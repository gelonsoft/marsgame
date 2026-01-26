import copy
import random
import torch
import threading
import time
import string
from replay import ReplayBuffer
from muzero import MuZeroNet
from mcts import MCTS
from experiment_manager import ExperimentManager
from env_all_actions import parallel_env
from observe_gamestate import observe
from evaluator import Evaluator
import os
from promotion_tournament import play_five_player_game
import random
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BackgroundTrainer:
    def __init__(self, args,obs_dim, action_dim):
        self.enable_train=not args.disable_train
        self.enable_promotion=args.enable_prom
        self.run_name=''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
        self.env = parallel_env()
        self.replay=None
        self.model=None
        self.optimizer=None
        if self.enable_train:
            self.replay = ReplayBuffer(
                capacity=500_000,           # disk capacity
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=DEVICE,
                directory="replay_disk",
                cache_size=1024             # RAM window (~270MB)
            )
            self.model = MuZeroNet(obs_dim, action_dim).to(DEVICE)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=1e-4,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        self.obs_dim=obs_dim
        self.action_dim=action_dim
        self.episode = 0
        self.win_counter = 0
        
        self.mcts = MCTS(self.model, action_dim, sims=32,discount=0.997)
        self.manager = ExperimentManager(os.path.join("runs","muzero"))

        if self.enable_train:
            self.manager.load_latest(self.model)

        self.running = True

        if self.enable_train:
            self.rollout_thread = threading.Thread(target=self.rollout_loop, daemon=True)
            self.train_thread = threading.Thread(target=self.train_loop, daemon=True)

        #self.evaluator = Evaluator(self.model, self.manager)
        #self.eval_thread = threading.Thread(target=self.evaluator.run, daemon=True)
        if self.enable_promotion:
            self.promotion_thread = threading.Thread(target=self.promotion_loop, daemon=True)

    def start(self):
        if self.enable_train:
            self.rollout_thread.start()
            self.train_thread.start()
        #self.eval_thread.start()
        if self.enable_promotion:
            self.promotion_thread.start()

        while True:
            time.sleep(60)

    def rollout_loop(self):
        replay_data={}
        while self.running:
            if len(self.replay) > 500_000:
                time.sleep(1)
                continue
            obs, infos, action_count, action_list,current_env_id = self.env.reset()
            self.episode += 1
            episode_reward = 0

            done = False
            with torch.no_grad():
                while not done:
                    action=0
                    obs_vec = torch.tensor(obs, dtype=torch.float32,device=DEVICE)
                    policy = torch.tensor(self.mcts.run(obs_vec,legal_actions=action_list,temperature=1.0,training=True,)).detach().cpu()
                    policy=policy.detach().cpu()
                    policy = policy + 1e-8               # avoid zeros
                    policy = policy / policy.sum()       # normalize
                    if action_count>0:

                        mask = self.env.get_action_mask(action_list)
                        masked_policy = policy * torch.tensor(mask, device=policy.device)
                        masked_policy_sum=masked_policy.sum()
                        if (masked_policy_sum.abs()>1e-18).any():
                            masked_policy = masked_policy / masked_policy.sum()
                        masked_policy2=torch.clamp(masked_policy,1e-18,1e6)
                        action = int(torch.multinomial(masked_policy2, 1).detach().cpu())
                        if action not in action_list:
                            action=random.choice(action_list)


                    next_obs, rewards, terms, action_count,action_list,all_rewards,curr_eid = self.env.step(action)#{action})

                    reward = float(max(-1.0, min(1.0, rewards)))
                    episode_reward += reward
                    done =terms

                    next_vec = torch.tensor(next_obs, dtype=torch.float32)
                    if curr_eid==current_env_id:
                        self.replay.add(obs_vec.cpu(), action, reward, next_vec, done, policy)
                    else:
                        replay_data[current_env_id]={
                            'obs':obs_vec.cpu(),
                            'action':action,
                            'reward':reward,
                            'done':done,
                            'policy':policy
                        }
                        if curr_eid in replay_data:
                            r=replay_data[curr_eid]
                            self.replay.add(r['obs'], r['action'], r['reward'], next_vec, r['done'], r['policy'])
                        current_env_id=curr_eid


                    obs = next_obs

                # Episode finished
                self.manager.log_metric("rollout/episode_reward", episode_reward)
                self.manager.log_metric("rollout/episode", self.episode)

    def promotion_loop(self):
        while True:
            time.sleep(30)  # every 5 minutes

            best_pool_files = self.manager.list_best_pool()
            original_last_agents = self.manager.list_last_agents()

            if len(best_pool_files) < 2 or len(last_agents) < 3:
                print("Not enough agents for promotion tournament.")
                continue

            # Step 1: sample agents
            pool_agents = random.sample(best_pool_files, 2)
            last_agents = random.sample(original_last_agents, 3)

            agent_files = pool_agents + last_agents

            models = []
            names = []

            for i in range(len(agent_files)):
                path=agent_files[i]
                model = MuZeroNet(self.obs_dim, self.action_dim).to(DEVICE)
                try:
                    model.load_state_dict(torch.load(path, map_location=DEVICE))
                except:
                    best_pool_files = self.manager.list_best_pool()
                    original_last_agents = self.manager.list_last_agents()
                    if i<2:
                        best_pool_files = self.manager.list_best_pool()
                        if len(best_pool_files)<2:
                            best_pool_files.append(original_last_agents[0])
                        path=random.choice(best_pool_files)
                        agent_files[i]=path
                        model.load_state_dict(torch.load(path, map_location=DEVICE))
                    else:
                        path = random.choice(original_last_agents)
                        agent_files[i]=path
                        model.load_state_dict(torch.load(path, map_location=DEVICE))

                model.eval()

                models.append(model)
                names.append(path)

            # Step 2: play 10 games
            total_rewards = {name: 0.0 for name in names}
            total_prewards = {name: 0.0 for name in names}
            total_places = {name: 0 for name in names}

            for g in range(10):
                rewards,prewards, vp = play_five_player_game(models,self.replay, DEVICE)
                print(f"Promotions rewards={rewards} prewards={prewards}")
                # rank by victory points
                sorted_agents = sorted(vp.items(), key=lambda x: -x[1])

                for place, (agent_id, score) in enumerate(sorted_agents):
                    name = names[int(agent_id)-1]
                    total_places[name] += place
                    total_rewards[name] += rewards[int(agent_id)]
                    total_prewards[name] += prewards[agent_id]

            # Step 3: choose top 2 by avg place
            avg_place = {k: total_places[k] / 10 for k in names}
            avg_reward = {os.path.basename(k): total_rewards[k] / 10 for k in names}

            ranked = sorted(names, key=lambda k: avg_place[k])
            avg_place = {os.path.basename(k): total_places[k] / 10 for k in names}
            winners = ranked[:2]

            # Step 4: save stats
            stats = {
                "candidates": [os.path.basename(name) for name in  names],
                "avg_place": avg_place,
                "avg_reward": avg_reward,
                "winners": [os.path.basename(name) for name in  winners],
            }

            self.manager.save_promotion_stats(stats)

            for name in names:
                self.manager.writer.add_scalar("promotion/avg_place/" + os.path.basename(name), avg_place[os.path.basename(name)], self.manager.global_step)
                self.manager.writer.add_scalar("promotion/avg_reward/" + os.path.basename(name), avg_reward[os.path.basename(name)], self.manager.global_step)

            # Step 5: update best pool
            for path in pool_agents:
                if path not in winners:
                    os.remove(path)

            for w in winners:
                src = w
                dst = os.path.join(self.manager.best_pool_dir, os.path.basename(w))
                if src!=dst:
                    torch.save(torch.load(src), dst)

            models=[]
            model=None
            # Keep only top 10 in pool
            pool_files = self.manager.list_best_pool()
            if len(pool_files) > 10:
                pool_files.sort(key=os.path.getmtime)
                for p in pool_files[:-10]:
                    os.remove(p)

            agent_files = self.manager.list_last_agents()
            if len(agent_files) > 10:
                agent_files.sort(key=os.path.getmtime)
                for p in agent_files[:-10]:
                    os.remove(p)

            print("Promotion complete. Winners:", winners)


    def train_loop(self):
        step = 0
        while self.running:
            if len(self.replay) < 256:
                time.sleep(1)
                continue

            obs, act, rew, next_obs, done, policy_target = self.replay.sample(256)
            policy_target = policy_target + 1e-8               # avoid zeros
            policy_target = policy_target / policy_target.sum(dim=1, keepdim=True)       # normalize
            obs = obs.to(DEVICE)
            next_obs = next_obs.to(DEVICE)
            rew = rew.to(DEVICE)

            policy_pred, value, h = self.model.initial_inference(obs)


            with torch.no_grad():
                _, next_value, _ = self.model.initial_inference(next_obs)
                target_value = rew + 0.997 * next_value.squeeze() * (1 - done.float())

            #loss = ((value.squeeze() - rew) ** 2).mean()
            value_loss = ((value.squeeze() - target_value) ** 2).mean()
            policy_target = torch.clamp(policy_target, 1e-8, 1.0)
            policy_target = policy_target / policy_target.sum(dim=1, keepdim=True)
            policy_pred = torch.clamp(policy_pred, 1e-8, 1.0)
            policy_pred = policy_pred / policy_pred.sum(dim=1, keepdim=True)
            policy_loss = -(policy_target * torch.log(policy_pred + 1e-8)).sum(dim=1).mean()
            loss = value_loss + policy_loss
            #loss = ((value.squeeze() - rew) ** 2).mean()
            # TensorBoard logging
            self.manager.log_metric("train/value_loss", loss.item())
            self.manager.log_metric("train/replay_size", len(self.replay))

            self.optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            loss.backward()
            self.optimizer.step()

            self.manager.step()

            step += 1

            if step % 100 == 0:
                print("Training step", step, "loss", loss.item())

            if step % 1000 == 0:
                self.replay.save()

            self.manager.autosave(self.model, f"agent_{self.run_name}_{step}")
