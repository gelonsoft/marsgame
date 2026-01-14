import random
import torch
import threading
import time
from replay import ReplayBuffer
from muzero import MuZeroNet
from mcts import MCTS
from experiment_manager import ExperimentManager
from env import parallel_env
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

    def start(self):
        self.rollout_thread.start()
        self.train_thread.start()
        self.eval_thread.start()

        while True:
            time.sleep(60)

    def rollout_loop(self):
        while self.running:
            obs, _ = self.env.reset()
            self.episode += 1
            episode_reward = 0

            done = False
            while not done:
                action_map={}
                for agent in ["1"]:
                    
                    obs_vec = torch.tensor(obs[agent], dtype=torch.float32,device=DEVICE)
                    policy = torch.tensor(self.mcts.run(obs_vec)).detach().cpu()
                    acts=list(self.env.action_lookup[agent].keys())
                    if len(acts)==0:
                        action_map[agent]=0
                    else:
                        policy=policy.detach().cpu()
                        mask = self.env.get_action_mask(agent)
                        masked_policy = policy * torch.tensor(mask, device=policy.device)
                        masked_policy_sum=masked_policy.sum()
                        if (masked_policy_sum.abs()>1e-18).any():
                            masked_policy = masked_policy / masked_policy.sum()
                        masked_policy2=torch.clamp(masked_policy,1e-18,1e6)
                        action = int(torch.multinomial(masked_policy2, 1).detach().cpu())
                        if action not in acts:
                            action_map[agent]=random.choice(acts)
                        else:
                            action_map[agent]=action
                for agent in ["2"]:
                    mask=self.env.get_action_mask(agent)
                    acts=list(self.env.action_lookup[agent].keys())
                    if len(acts)==0:
                        acts=[0]
                    action_map[agent]=random.choice(acts)

                # obs_vec = torch.tensor(obs["1"], dtype=torch.float32).to(DEVICE)
                # policy = torch.tensor(self.mcts.run(obs_vec)).detach().cpu()
                # mask = self.env.get_action_mask("1")
                # masked_policy = policy * torch.tensor(mask, device=policy.device)
                # masked_policy = masked_policy / masked_policy.sum()

                # action = int(torch.multinomial(masked_policy, 1))

                next_obs, rewards, terms, truncs, infos = self.env.step(action_map)#{"1": action, "2": action})

                reward = rewards["1"]
                episode_reward += reward
                done = any(terms.values())

                next_vec = torch.tensor(next_obs["1"], dtype=torch.float32)

                self.replay.add(obs_vec.cpu(), action, reward, next_vec, done)
                obs = next_obs

            # Episode finished
            self.manager.log_metric("rollout/episode_reward", episode_reward)
            self.manager.log_metric("rollout/episode", self.episode)

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
