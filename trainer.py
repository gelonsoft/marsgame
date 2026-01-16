import copy
import random
import torch
import threading
import time
from promotion import play_match
from replay import ReplayBuffer
from muzero import MuZeroNet
from mcts import MCTS
from experiment_manager import ExperimentManager
from env_all_actions import parallel_env
from observe_gamestate import observe
from evaluator import Evaluator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BackgroundTrainer:
    def __init__(self, obs_dim, action_dim):
        self.env = parallel_env()
        self.replay = ReplayBuffer(200_000)

        self.episode = 0
        self.win_counter = 0

        self.model = MuZeroNet(obs_dim, action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.mcts = MCTS(self.model, action_dim)
        self.manager = ExperimentManager("runs/muzero")

        self.manager.load_latest(self.model)

        self.running = True

        self.rollout_thread = threading.Thread(target=self.rollout_loop, daemon=True)
        self.train_thread = threading.Thread(target=self.train_loop, daemon=True)

        self.evaluator = Evaluator(self.model, self.manager)
        self.eval_thread = threading.Thread(target=self.evaluator.run, daemon=True)
        self.promotion_thread = threading.Thread(target=self.promotion_loop, daemon=True)

    def start(self):
        self.rollout_thread.start()
        self.train_thread.start()
        self.eval_thread.start()
        #self.promotion_thread.start()

        while True:
            time.sleep(60)

    def rollout_loop(self):
        while self.running:
            obs, infos, action_count, action_list = self.env.reset()
            self.episode += 1
            episode_reward = 0

            done = False
            while not done:
                action=0
                for agent in ["1"]:
                    
                    obs_vec = torch.tensor(obs, dtype=torch.float32,device=DEVICE)
                    policy = torch.tensor(self.mcts.run(obs_vec)).detach().cpu()
                    if action_count>0:
                        policy=policy.detach().cpu()
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

                reward = rewards
                episode_reward += reward
                done =terms

                next_vec = torch.tensor(next_obs, dtype=torch.float32)

                self.replay.add(obs_vec.cpu(), action, reward, next_vec, done)
                obs = next_obs

            # Episode finished
            self.manager.log_metric("rollout/episode_reward", episode_reward)
            self.manager.log_metric("rollout/episode", self.episode)

    def promotion_loop(self):
        while True:
            # wait until model has trained a bit
            if len(self.replay) < 5000:
                time.sleep(30)
                continue

            # snapshot current agent
            agent_copy = copy.deepcopy(self.model).cpu()
            name = f"agent_{self.agent_id}"
            self.agent_id += 1

            # play against best agents
            for opponent_name, opponent_model in self.best_pool:
                result = play_match(agent_copy, opponent_model)

                self.manager.elo.update(name, opponent_name, result)

            # compute Elo
            rating = self.manager.elo.get(name)

            # REGISTER AGENT  ← this is what was missing
            self.manager.register(name, rating)

            # maintain best-5 pool
            self.best_pool.append((name, agent_copy))
            self.best_pool = sorted(
                self.best_pool,
                key=lambda x: self.manager.elo.get(x[0]),
                reverse=True
            )[:5]

            # save promoted agent
            self.manager.save(agent_copy, name)

            print("Promoted:", name, "Elo:", rating)
            time.sleep(300)   # every 5 minutes


    def train_loop(self):
        step = 0
        while self.running:
            if len(self.replay) < 1024:
                time.sleep(1)
                continue

            obs, act, rew, next_obs, done = self.replay.sample(256)

            obs = obs.to(DEVICE)
            next_obs = next_obs.to(DEVICE)
            rew = rew.to(DEVICE)

            policy, value, h = self.model.initial_inference(obs)

            loss = ((value.squeeze() - rew) ** 2).mean()
            # TensorBoard logging
            self.manager.log_metric("train/value_loss", loss.item())
            self.manager.log_metric("train/replay_size", len(self.replay))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.manager.step()

            step += 1

            if step % 100 == 0:
                print("Training step", step, "loss", loss.item())

            self.manager.autosave(self.model, f"agent_{step}")
