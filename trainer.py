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
import numpy as np
from legal_actions_decoder import (
    get_legal_actions_from_obs
)
from rollout import ParallelRolloutManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RunningMeanStd:
    """
    Running mean and standard deviation for normalizing value targets.
    Essential for stable value learning when rewards have high variance.
    """
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x):
        """Update running statistics with new batch"""
        x = np.array(x).flatten()
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # Welford's online algorithm
        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x):
        """Normalize values using running statistics"""
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)
    
    def denormalize(self, x):
        """Denormalize values back to original scale"""
        return x * np.sqrt(self.var) + self.mean
    
class BackgroundTrainer:
    def __init__(self, args, obs_dim, action_dim):
        self.enable_train = not args.disable_train
        self.enable_promotion = args.enable_prom
        self.run_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
        self.env = parallel_env()
        self.replay = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        self.mcts = MCTS(
            self.model, 
            action_dim, 
            sims=100,
            c_puct=1.5,
            discount=0.997,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25
        )
        self.manager = ExperimentManager(os.path.join("runs", "muzero"))

        if self.enable_train:
            self.replay = ReplayBuffer(
                capacity=500_000,
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=DEVICE,
                directory="replay_disk",
                cache_size=2048
            )
            self.model = MuZeroNet(obs_dim, action_dim).to(DEVICE)
            self.value_normalizer = RunningMeanStd()
            self.reward_normalizer = RunningMeanStd()
            self.parallel_rollout = ParallelRolloutManager(
                model=self.model,
                replay_buffer=self.replay,
                manager=self.manager,
                num_workers=6,
                action_dim=action_dim,
                obs_dim=obs_dim,
                mcts_config={
                    'sims': 100,
                    'c_puct': 1.5,
                    'discount': 0.997,
                    'dirichlet_alpha': 0.3,
                    'dirichlet_frac': 0.25
                }
            )

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=1e-4,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=50000, eta_min=1e-6
            )
            
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.episode = 0
        self.win_counter = 0
        
        # IMPROVED: MCTS will extract legal actions from observations


        if self.enable_train:
            self.manager.load_latest(self.model)

        self.running = True
        
        self.train_step = 0
        self.policy_loss_ema = 0.0
        self.value_loss_ema = 0.0
        self.alpha_ema = 0.98

        if self.enable_train:

            self.train_thread = threading.Thread(target=self.train_loop, daemon=True)

        if self.enable_promotion:
            self.promotion_thread = threading.Thread(target=self.promotion_loop, daemon=True)

    def start(self):
        if self.enable_train:
            self.parallel_rollout.start()
            # Wait for initial buffer fill
            self.parallel_rollout.wait_for_buffer_filled(min_size=512)
            # Start training
            self.train_thread.start()
            # Start continuous rollout
            self.rollout_thread = threading.Thread(
                target=self.parallel_rollout.rollout_loop,
                daemon=True
            )
            self.rollout_thread.start()

        if self.enable_promotion:
            self.promotion_thread.start()

        while True:
            time.sleep(60)

    def promotion_loop(self):
        while True:
            time.sleep(30)

            best_pool_files = self.manager.list_best_pool()
            original_last_agents = self.manager.list_last_agents()

            if len(best_pool_files) < 2 or len(original_last_agents) < 3:
                print("Not enough agents for promotion tournament.")
                continue

            pool_agents = random.sample(best_pool_files, 2)
            last_agents = random.sample(original_last_agents, 3)

            agent_files = pool_agents + last_agents

            models = []
            names = []

            for i in range(len(agent_files)):
                path = agent_files[i]
                model = MuZeroNet(self.obs_dim, self.action_dim).to(DEVICE)
                try:
                    model.load_state_dict(torch.load(path, map_location=DEVICE))
                except:
                    best_pool_files = self.manager.list_best_pool()
                    original_last_agents = self.manager.list_last_agents()
                    if i < 2:
                        best_pool_files = self.manager.list_best_pool()
                        if len(best_pool_files) < 2:
                            best_pool_files.append(original_last_agents[0])
                        path = random.choice(best_pool_files)
                        agent_files[i] = path
                        model.load_state_dict(torch.load(path, map_location=DEVICE))
                    else:
                        path = random.choice(original_last_agents)
                        agent_files[i] = path
                        model.load_state_dict(torch.load(path, map_location=DEVICE))

                model.eval()
                models.append(model)
                names.append(path)

            total_rewards = {name: 0.0 for name in names}
            total_prewards = {name: 0.0 for name in names}
            total_places = {name: 0 for name in names}

            for g in range(10):
                rewards, prewards, vp = play_five_player_game(models, self.replay, DEVICE)
                print(f"Promotions rewards={rewards} prewards={prewards}")
                
                sorted_agents = sorted(vp.items(), key=lambda x: -x[1])

                for place, (agent_id, score) in enumerate(sorted_agents):
                    name = names[int(agent_id) - 1]
                    total_places[name] += place
                    total_rewards[name] += rewards[int(agent_id)]
                    total_prewards[name] += prewards[agent_id]

            avg_place = {k: total_places[k] / 10 for k in names}
            avg_reward = {os.path.basename(k): total_rewards[k] / 10 for k in names}

            ranked = sorted(names, key=lambda k: avg_place[k])
            avg_place = {os.path.basename(k): total_places[k] / 10 for k in names}
            winners = ranked[:2]

            stats = {
                "candidates": [os.path.basename(name) for name in names],
                "avg_place": avg_place,
                "avg_reward": avg_reward,
                "winners": [os.path.basename(name) for name in winners],
            }

            self.manager.save_promotion_stats(stats)

            for name in names:
                self.manager.writer.add_scalar("promotion/avg_place/" + os.path.basename(name), avg_place[os.path.basename(name)], self.manager.global_step)
                self.manager.writer.add_scalar("promotion/avg_reward/" + os.path.basename(name), avg_reward[os.path.basename(name)], self.manager.global_step)

            for path in pool_agents:
                if path not in winners:
                    os.remove(path)

            for w in winners:
                src = w
                dst = os.path.join(self.manager.best_pool_dir, os.path.basename(w))
                if src != dst:
                    torch.save(torch.load(src), dst)

            models = []
            model = None
            
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
        accumulation_steps = 2
        while self.running:
            if not self.enable_train or self.replay is None or self.model is None or self.optimizer is None or self.scheduler is None:
                raise Exception("Bad 2")
            if len(self.replay) < 512:
                time.sleep(1)
                continue

            batch_size = 256
            obs, act, rew, next_obs, done, policy_target = self.replay.sample(batch_size)
            
            # Normalize policy target
            policy_target = policy_target + 1e-8
            policy_target = policy_target / policy_target.sum(dim=1, keepdim=True)
            
            obs = obs.to(DEVICE)
            next_obs = next_obs.to(DEVICE)
            rew = rew.to(DEVICE)
            done = done.to(DEVICE)
            policy_target = policy_target.to(DEVICE)

            # NEW: Apply legal action masking to policy targets
            # Extract legal actions from observations
            obs_np = obs.detach().cpu().numpy()
            legal_actions_batch = []
            for i in range(batch_size):
                legal_actions = get_legal_actions_from_obs(obs_np[i])
                legal_actions_batch.append(legal_actions)
            
            # Apply mask to policy targets
            for i in range(batch_size):
                legal_actions = legal_actions_batch[i]
                if len(legal_actions) > 0:
                    # Zero out illegal actions
                    mask = torch.zeros(self.action_dim, device=DEVICE)
                    mask[legal_actions] = 1.0
                    
                    policy_target[i] = policy_target[i] * mask
                    
                    # Renormalize
                    total = policy_target[i].sum()
                    if total > 1e-8:
                        policy_target[i] = policy_target[i] / total
                    else:
                        # Uniform over legal actions
                        policy_target[i] = mask / len(legal_actions)

            # Forward pass
            policy_pred, value, h = self.model.initial_inference(obs)

            # NEW: Apply legal action masking to predictions
            for i in range(batch_size):
                legal_actions = legal_actions_batch[i]
                if len(legal_actions) > 0:
                    mask = torch.zeros(self.action_dim, device=DEVICE)
                    mask[legal_actions] = 1.0
                    
                    policy_pred[i] = policy_pred[i] * mask
                    
                    # Renormalize with softmax over legal actions
                    policy_pred[i] = torch.softmax(policy_pred[i], dim=-1)

            # Compute target value with TD learning
            with torch.no_grad():
                _, next_value, _ = self.model.initial_inference(next_obs)
                
                # Denormalize next_value for TD computation
                next_value_denorm = self.value_normalizer.denormalize(next_value.detach().cpu().numpy())
                
                # Compute TD target in original scale
                rew_np = rew.detach().cpu().numpy()
                done_np = done.detach().cpu().numpy()
                target_value_denorm = rew_np + 0.997 * next_value_denorm.squeeze() * (1 - done_np)
                
                # Update normalizer statistics
                self.value_normalizer.update(target_value_denorm)
                
                # Normalize target for training
                target_value_norm = self.value_normalizer.normalize(target_value_denorm)
                target_value = torch.tensor(target_value_norm, dtype=torch.float32, device=DEVICE)

            # Also normalize the predicted value for loss computation
            value_denorm = self.value_normalizer.denormalize(value.detach().cpu().numpy())
            value_norm = self.value_normalizer.normalize(value_denorm)
            value = torch.tensor(value_norm, dtype=torch.float32, device=DEVICE)

            # Huber loss for value (more robust to outliers)
            value_loss = torch.nn.functional.smooth_l1_loss(value.squeeze(), target_value)
            
            # Better policy loss with label smoothing
            policy_target_smooth = policy_target * 0.95 + 0.05 / self.action_dim
            policy_target_smooth = torch.clamp(policy_target_smooth, 1e-8, 1.0)
            policy_target_smooth = policy_target_smooth / policy_target_smooth.sum(dim=1, keepdim=True)
            
            policy_pred = torch.clamp(policy_pred, 1e-8, 1.0)
            policy_pred = policy_pred / policy_pred.sum(dim=1, keepdim=True)
            
            # Cross-entropy loss
            legal_mask = torch.zeros(batch_size, self.action_dim, device=DEVICE)
            for i in range(batch_size):
                legal_actions = legal_actions_batch[i]
                if len(legal_actions) > 0:
                    legal_mask[i, legal_actions] = 1.0

            # Mask both prediction and target
            policy_pred_masked = policy_pred * legal_mask
            policy_target_masked = policy_target * legal_mask

            # Renormalize
            policy_pred_masked = policy_pred_masked / (policy_pred_masked.sum(dim=1, keepdim=True) + 1e-8)
            policy_target_masked = policy_target_masked / (policy_target_masked.sum(dim=1, keepdim=True) + 1e-8)

            # KL divergence loss (more stable than cross-entropy for continuous distributions)
            policy_loss = torch.nn.functional.kl_div(
                torch.log(policy_pred_masked + 1e-8),
                policy_target_masked,
                reduction='batchmean'
            )

            # Add entropy bonus to encourage exploration
            # This prevents premature convergence to deterministic policies
            num_legal_per_sample = legal_mask.sum(dim=1, keepdim=True)
            entropy_per_action = -(policy_pred_masked * torch.log(policy_pred_masked + 1e-8))
            entropy = (entropy_per_action * legal_mask).sum(dim=1).mean()

            # Adaptive entropy weight - decay over time
            entropy_weight = 0.02 * max(0.1, 1.0 - step / 30000)
            policy_loss = policy_loss - entropy_weight * entropy
            
            # Dynamic loss weighting
            value_weight = 1.0
            policy_weight = 1.0 if step < 5000 else 0.8
            
            loss = value_weight * value_loss + policy_weight * policy_loss
            
            # Add L2 regularization on hidden states
            if step % 10 == 0:
                reg_loss = 0.0001 * torch.norm(h, p=2)
                loss = loss + reg_loss

            # Backward pass with gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                # Compute gradient norm for monitoring
                total_norm = 0.0
                param_count = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                total_norm = total_norm ** 0.5
                
                # Log every 100 steps
                if step % 100 == 0:
                    self.manager.log_metric("train/grad_norm", total_norm)
                    self.manager.log_metric("train/grad_norm_per_param", total_norm / max(param_count, 1))
                
                # Adaptive gradient clipping based on norm
                # If gradients are exploding (>10), use stricter clipping
                max_grad_norm = 0.5 if total_norm > 10.0 else 1.0
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            # Update EMA metrics
            self.policy_loss_ema = self.alpha_ema * self.policy_loss_ema + (1 - self.alpha_ema) * policy_loss.item()
            self.value_loss_ema = self.alpha_ema * self.value_loss_ema + (1 - self.alpha_ema) * value_loss.item()

            # Logging
            self.manager.log_metric("train/value_loss", value_loss.item())
            self.manager.log_metric("train/policy_loss", policy_loss.item())
            self.manager.log_metric("train/total_loss", loss.item() * accumulation_steps)
            self.manager.log_metric("train/replay_size", len(self.replay))
            self.manager.log_metric("train/learning_rate", self.optimizer.param_groups[0]['lr'])
            
            # NEW: Log legal actions statistics
            if step % 100 == 0:
                avg_legal_actions = np.mean([len(la) for la in legal_actions_batch])
                self.manager.log_metric("train/avg_legal_actions", avg_legal_actions)
            
            if step % 10 == 0:
                self.manager.log_metric("train/value_loss_ema", self.value_loss_ema)
                self.manager.log_metric("train/policy_loss_ema", self.policy_loss_ema)

            if step % 100 == 0:
                # Log more detailed metrics
                avg_legal_actions = np.mean([len(la) for la in legal_actions_batch])
                self.manager.log_metric("train/avg_legal_actions", avg_legal_actions)
                
                # Log value statistics
                with torch.no_grad():
                    value_mean = value.mean().item()
                    value_std = value.std().item()
                    target_mean = target_value.mean().item()
                    target_std = target_value.std().item()
                    
                    self.manager.log_metric("train/value_mean", value_mean)
                    self.manager.log_metric("train/value_std", value_std)
                    self.manager.log_metric("train/target_mean", target_mean)
                    self.manager.log_metric("train/target_std", target_std)
                
                # Log policy statistics
                with torch.no_grad():
                    policy_entropy = -(policy_pred * torch.log(policy_pred + 1e-8)).sum(dim=1).mean().item()
                    self.manager.log_metric("train/policy_entropy", policy_entropy)
                    
                    # Max probability in predictions (low = more exploration)
                    max_probs = policy_pred.max(dim=1)[0].mean().item()
                    self.manager.log_metric("train/policy_max_prob", max_probs)
            self.manager.step()
            step += 1
            self.train_step = step

            if step % 100 == 0:
                print(f"Training step {step}, loss {loss.item() * accumulation_steps:.4f}, "
                      f"value_loss {value_loss.item():.4f}, policy_loss {policy_loss.item():.4f}, "
                      f"lr {self.optimizer.param_groups[0]['lr']:.6f}")

            if step % 1000 == 0:
                self.replay.save()

            self.manager.autosave(self.model, f"agent_{self.run_name}_{step}")