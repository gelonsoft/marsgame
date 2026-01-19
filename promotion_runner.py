import argparse
import threading
import time
import traceback
import random
import os
import torch

from muzero import MuZeroNet
from experiment_manager import ExperimentManager
from promotion_tournament import play_five_player_game


class PromotionWorker(threading.Thread):
    def __init__(self, worker_id, manager, obs_dim, action_dim, device):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.manager = manager
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.running = True

    def run(self):
        while self.running:
            try:
                self.run_promotion_cycle()
            except Exception as e:
                print(f"[Worker {self.worker_id}] Promotion thread crashed. Restarting.")
                traceback.print_exc()
                time.sleep(5)  # backoff before restart

    def run_promotion_cycle(self):
        print(f"[Worker {self.worker_id}] Starting promotion cycle")


        best_pool_files = self.manager.list_best_pool()
        original_last_agents = self.manager.list_last_agents()

        if len(best_pool_files) < 2 or len(original_last_agents) < 3:
            print(f"[Worker {self.worker_id}] Not enough agents for promotion tournament.")
            return

        # Step 1: sample agents
        pool_agents = random.sample(best_pool_files, 2)
        last_agents = random.sample(original_last_agents, 3)
        agent_files = pool_agents + last_agents

        models = []
        names = []

        for path in agent_files:
            model = MuZeroNet(self.obs_dim, self.action_dim).to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()

            models.append(model)
            names.append(path)

        # Step 2: play games
        total_rewards = {name: 0.0 for name in names}
        total_prewards = {name: 0.0 for name in names}
        total_places = {name: 0 for name in names}

        for g in range(10):
            rewards, prewards, vp = play_five_player_game(models, None, self.device)

            sorted_agents = sorted(vp.items(), key=lambda x: -x[1])

            for place, (agent_id, score) in enumerate(sorted_agents):
                name = names[int(agent_id) - 1]
                total_places[name] += place
                total_rewards[name] += rewards[int(agent_id)]
                total_prewards[name] += prewards[agent_id]

        # Step 3: ranking
        avg_place = {k: total_places[k] / 10 for k in names}
        avg_reward = {os.path.basename(k): total_rewards[k] / 10 for k in names}

        ranked = sorted(names, key=lambda k: avg_place[k])
        winners = ranked[:2]

        # Step 4: save stats
        stats = {
            "candidates": [os.path.basename(name) for name in names],
            "avg_place": {os.path.basename(k): v for k, v in avg_place.items()},
            "avg_reward": avg_reward,
            "winners": [os.path.basename(name) for name in winners],
        }

        self.manager.save_promotion_stats(stats)

        # Step 5: update best pool
        for path in pool_agents:
            if path not in winners and os.path.exists(path) and len(self.manager.list_best_pool())>10:
                os.remove(path)

        for w in winners:
            os.utime(w, None)
            dst = os.path.join(self.manager.best_pool_dir, os.path.basename(w))
            if w != dst:
                torch.save(torch.load(w), dst)

        # Keep only top 10
        pool_files = self.manager.list_best_pool()
        if len(pool_files) > 10:
            pool_files.sort(key=os.path.getmtime)
            for p in pool_files[:-10]:
                os.remove(p)

        agent_files = self.manager.list_last_agents()
        if len(agent_files) > 5:
            agent_files.sort(key=os.path.getmtime)
            for p in agent_files[:-5]:
                os.remove(p)

        print(f"[Worker {self.worker_id}] Promotion complete. Winners: {winners}")


class PromotionSupervisor:
    def __init__(self, num_threads, obs_dim, action_dim, device):
        self.num_threads = num_threads
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.manager = ExperimentManager(os.path.join("runs", "muzero"))
        self.workers = []

    def start(self):
        print(f"Starting promotion service with {self.num_threads} threads on device={self.device}")

        for i in range(self.num_threads):
            worker = PromotionWorker(
                worker_id=i,
                manager=self.manager,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                device=self.device
            )
            worker.start()
            self.workers.append(worker)

        # supervisor loop
        while True:
            time.sleep(60)
            self.monitor_workers()

    def monitor_workers(self):
        for i, worker in enumerate(self.workers):
            if not worker.is_alive():
                print(f"[Supervisor] Restarting dead worker {i}")
                new_worker = PromotionWorker(
                    worker_id=i,
                    manager=self.manager,
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    device=self.device
                )
                new_worker.start()
                self.workers[i] = new_worker


def main():
    parser = argparse.ArgumentParser(description="Promotion tournament service")
    parser.add_argument("--threads", type=int, default=3, help="Number of promotion threads")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device: cpu or cuda")

    args = parser.parse_args()

    device = torch.device(args.device)

    # read obs/action dims from latest agent
    manager = ExperimentManager(os.path.join("runs", "muzero"))
    last_agents = manager.list_last_agents()

    if not last_agents:
        raise RuntimeError("No agents found. Cannot infer model dimensions.")

    # Load one agent to infer dimensions
    tmp_model = torch.load(last_agents[-1], map_location="cpu")
    obs_dim = tmp_model["representation.0.weight"].shape[1]
    action_dim = tmp_model["policy_head.weight"].shape[0]

    supervisor = PromotionSupervisor(
        num_threads=args.threads,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )

    supervisor.start()


if __name__ == "__main__":
    main()
