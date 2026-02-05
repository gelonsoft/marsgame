"""
FIXED Parallel Rollout Implementation - WITH METADATA SUPPORT
==============================================================

Enhanced to provide all metadata needed for the improved replay buffer:
- Generation number (for curriculum + prioritization)
- Terraform completion % (for multi-critic)
- TD error (for prioritization)
- Auxiliary targets (for self-supervised learning)
- Head masks (for mask-aware entropy)
"""

import copy
import random
import torch
import threading
import time
import queue
import numpy as np
from typing import List, Dict, Tuple, Optional
from env_all_actions import parallel_env
from mcts import MCTS
from factored_actions import FactoredActionEncoder, FACTORED_ACTION_DIMS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_auxiliary_targets(player_state_before, player_state_after):
    """
    Extract auxiliary prediction targets from game state.
    
    Returns:
        dict with keys: tr_delta, production, vp_delta, can_play_card, resource_sufficient
    """
    try:
        player_before = player_state_before.get('thisPlayer', {})
        player_after = player_state_after.get('thisPlayer', {})
        
        # TR delta
        tr_before = player_before.get('terraformRating', 20)
        tr_after = player_after.get('terraformRating', 20)
        tr_delta = float(tr_after - tr_before)
        
        # Production (6 types)
        prod_keys = [
            'megaCreditProduction', 'steelProduction', 'titaniumProduction',
            'plantProduction', 'energyProduction', 'heatProduction'
        ]
        production = np.array([
            player_after.get(k, 0) for k in prod_keys
        ], dtype=np.float32)
        
        # VP delta
        vp_before = player_before.get('victoryPointsBreakdown', {}).get('total', 0)
        vp_after = player_after.get('victoryPointsBreakdown', {}).get('total', 0)
        vp_delta = float(vp_after - vp_before)
        
        # Can play card (binary)
        hand_size = len(player_after.get('cardsInHand', []))
        mc = player_after.get('megaCredits', 0)
        can_play_card = float(hand_size > 0 and mc >= 5)
        
        # Resource sufficient (has resources to pay for cheapest card)
        resource_sufficient = float(mc >= 3)
        
        return {
            'tr_delta': tr_delta,
            'production': production,
            'vp_delta': vp_delta,
            'can_play_card': can_play_card,
            'resource_sufficient': resource_sufficient
        }
    except Exception as e:
        # Return defaults on error
        return {
            'tr_delta': 0.0,
            'production': np.zeros(6, dtype=np.float32),
            'vp_delta': 0.0,
            'can_play_card': 0.0,
            'resource_sufficient': 0.0
        }


def extract_game_metadata(env, player_state):
    """
    Extract metadata for prioritized replay and multi-critic.
    
    Returns:
        dict with: generation, terraform_pct, rounds_left
    """
    try:
        game_state = player_state.get('game', {})
        
        # Generation number (1-14 typically)
        generation = game_state.get('generation', 1)
        
        # Terraform completion %
        oxygen = game_state.get('oxygenLevel', 0)
        temp = game_state.get('temperature', -30)
        oceans = game_state.get('oceans', 0)
        
        # Calculate completion (max: oxygen=14, temp=8, oceans=9)
        oxygen_pct = min(oxygen / 14.0, 1.0)
        temp_pct = min((temp + 30) / 38.0, 1.0)  # -30 to +8 = 38 steps
        ocean_pct = min(oceans / 9.0, 1.0)
        terraform_pct = (oxygen_pct + temp_pct + ocean_pct) / 3.0
        
        # Rounds left (estimate)
        rounds_left = max(14 - generation, 0)
        
        return {
            'generation': generation,
            'terraform_pct': terraform_pct,
            'rounds_left': rounds_left,
            'terraform_complete': (oxygen >= 14 and temp >= 8 and oceans >= 9)
        }
    except Exception as e:
        return {
            'generation': 1,
            'terraform_pct': 0.0,
            'rounds_left': 14,
            'terraform_complete': False
        }


class RolloutWorker:
    """
    Enhanced rollout worker that collects metadata for improved replay buffer.
    """
    
    def __init__(
        self,
        worker_id: int,
        model,
        action_dim: int,
        obs_dim: int,
        mcts_config: dict,
        worker_queue: queue.Queue,
        lock: threading.Lock,
        stop_event: threading.Event
    ):
        self.worker_id = worker_id
        self.model = model
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.worker_queue = worker_queue
        self.lock = lock
        self.stop_event = stop_event
        
        self.env = parallel_env()
        self.mcts = MCTS(model=model, action_dim=action_dim, **mcts_config)
        
        self.episodes_completed = 0
        self.total_steps = 0
        self.factored_encoder = FactoredActionEncoder()
        
    def run_episode(self, episode_num: int, max_steps: int) -> Dict:
        """
        Run episode with full metadata collection.
        
        Returns:
            Dict with:
            - transitions: list of tuples (obs, action, reward, next_obs, done, policy, metadata)
            - episode_reward: float
            - step_count: int
        """
        transitions = []
        
        try:
            obs, infos, action_count, action_list, current_env_id = self.env.reset()
            
            # Track previous state for auxiliary targets
            prev_player_state = None
            
            # Initialize factored action data
            self.last_head_masks = None
            self.last_cond_obj = None
            self.last_cond_pay = None
            self.last_cond_extra = None
            self.last_factored_legal = None
            
            episode_reward = 0
            step_count = 0
            done = False
            
            # Dynamic temperature
            if episode_num < 2000:
                base_temperature = 1.5
            elif episode_num < 10000:
                base_temperature = 1.0
            else:
                base_temperature = max(0.25, 1.0 - (episode_num - 10000) / 40000)
            
            with torch.no_grad():
                while not done and step_count < max_steps:
                    obs_vec = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                    temperature = max(0.5, base_temperature - (step_count / max_steps) * 0.5)
                    
                    # Run MCTS
                    policy = torch.tensor(
                        self.mcts.run(
                            obs_vec,
                            legal_actions=action_list,
                            temperature=temperature,
                            training=True,
                            head_masks=self.last_head_masks,
                            cond_masks_obj_by_type=self.last_cond_obj,
                            cond_masks_pay_by_obj=self.last_cond_pay,
                            cond_masks_extra_by_pay=self.last_cond_extra,
                            factored_legal=self.last_factored_legal
                        )
                    ).detach().cpu()
                    
                    policy = policy + 1e-8
                    policy = policy / policy.sum()
                    
                    # Select action
                    action = 0
                    if action_count > 0:
                        legal_actions = action_list
                        mask = np.zeros(self.action_dim)
                        mask[legal_actions] = 1.0
                        
                        masked_policy = policy.numpy() * mask
                        masked_policy_sum = masked_policy.sum()
                        
                        if masked_policy_sum > 1e-18:
                            masked_policy = masked_policy / masked_policy_sum
                        else:
                            masked_policy = mask / len(legal_actions)
                        
                        masked_policy = np.clip(masked_policy, 1e-18, 1e6)
                        
                        try:
                            action = int(np.random.choice(self.action_dim, p=masked_policy))
                        except:
                            action = random.choice(legal_actions)
                        
                        if action not in legal_actions:
                            action = random.choice(legal_actions)
                    
                    # Get current player state BEFORE action
                    current_player_state = None
                    try:
                        current_env = self.env.envs[current_env_id]
                        current_player_state = copy.deepcopy(current_env.player_states)
                    except:
                        pass
                    
                    # Take action
                    step_result = self.env.step(action)
                    
                    # Unpack based on return length (handle both old and new env)
                    if len(step_result) == 12:
                        (next_obs, rewards, terms, action_count, action_list, 
                         all_rewards, curr_eid, head_masks, cond_obj, cond_pay, 
                         cond_extra, factored_legal) = step_result
                    else:
                        # Fallback for old env without factored data
                        next_obs, rewards, terms, action_count, action_list, all_rewards, curr_eid = step_result[:7]
                        head_masks = cond_obj = cond_pay = cond_extra = factored_legal = None
                    
                    # Store factored data for next MCTS call
                    self.last_head_masks = head_masks
                    self.last_cond_obj = cond_obj
                    self.last_cond_pay = cond_pay
                    self.last_cond_extra = cond_extra
                    self.last_factored_legal = factored_legal
                    
                    # Get player state AFTER action
                    next_player_state = None
                    try:
                        next_env = self.env.envs[curr_eid]
                        next_player_state = copy.deepcopy(next_env.player_states)
                    except:
                        pass
                    
                    # Extract metadata
                    game_metadata = extract_game_metadata(self.env, next_player_state or {})
                    
                    # Extract auxiliary targets
                    if current_player_state and next_player_state:
                        aux_targets = extract_auxiliary_targets(current_player_state, next_player_state)
                    else:
                        aux_targets = extract_auxiliary_targets({}, {})
                    
                    # Clip reward
                    reward = float(max(-2.0, min(2.0, rewards)))
                    episode_reward += reward
                    done = terms
                    step_count += 1
                    
                    next_vec = torch.tensor(next_obs, dtype=torch.float32)
                    
                    # Create metadata dict
                    metadata = {
                        'generation': game_metadata['generation'],
                        'terraform_pct': game_metadata['terraform_pct'],
                        'rounds_left': game_metadata['rounds_left'],
                        'aux_targets': aux_targets,
                        'head_masks': head_masks,
                        'td_error': 1.0,  # Will be updated after value prediction
                    }
                    
                    # Store transition with metadata
                    if curr_eid == current_env_id:
                        transitions.append((
                            obs_vec.cpu(),
                            action,
                            reward,
                            next_vec,
                            done,
                            policy,
                            metadata
                        ))
                    
                    # Update for next step
                    obs = next_obs
                    current_env_id = curr_eid
                    prev_player_state = next_player_state
                    
                    if done:
                        break
            
            self.episodes_completed += 1
            self.total_steps += step_count
            
            return {
                'worker_id': self.worker_id,
                'episode_num': episode_num,
                'transitions': transitions,
                'episode_reward': episode_reward,
                'step_count': step_count
            }
            
        except Exception as e:
            print(f"Worker {self.worker_id} error in episode: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'worker_id': self.worker_id,
                'episode_num': episode_num,
                'transitions': [],
                'episode_reward': 0.0,
                'step_count': 0
            }
    
    def run(self):
        """Worker thread main loop"""
        while not self.stop_event.is_set():
            try:
                task = self.worker_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    break
                
                episode_num, max_steps, result_queue = task
                result = self.run_episode(episode_num, max_steps)
                result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")


class ParallelRolloutManager:
    """
    Enhanced parallel rollout manager with metadata support.
    """
    
    def __init__(
        self,
        model,
        replay_buffer,
        manager,
        num_workers=6,
        action_dim=64,
        obs_dim=512,
        mcts_config=None
    ):
        self.model = model
        self.replay = replay_buffer
        self.manager = manager
        self.num_workers = num_workers
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        
        if mcts_config is None:
            mcts_config = {
                'sims': 100,
                'c_puct': 1.5,
                'discount': 0.997,
                'dirichlet_alpha': 0.3,
                'dirichlet_frac': 0.25
            }
        self.mcts_config = mcts_config
        
        # Threading
        self.task_queue = queue.Queue(maxsize=num_workers * 4)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
        # Workers
        self.workers = []
        self.worker_threads = []
        
        # Statistics
        self.episode = 0
        self.worker_stats = {i: {'episodes': 0, 'steps': 0} for i in range(num_workers)}
        self.total_transitions_collected = 0
        
    def start(self):
        """Start all worker threads"""
        print(f"Starting {self.num_workers} rollout workers...")
        
        for worker_id in range(self.num_workers):
            worker = RolloutWorker(
                worker_id=worker_id,
                model=self.model,
                action_dim=self.action_dim,
                obs_dim=self.obs_dim,
                mcts_config=self.mcts_config,
                worker_queue=self.task_queue,
                lock=self.lock,
                stop_event=self.stop_event
            )
            
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            
            self.workers.append(worker)
            self.worker_threads.append(thread)
        
        print(f"✓ {self.num_workers} workers started")
    
    def submit_episodes(self, count: int, max_steps: int):
        """Submit episode tasks to workers"""
        for _ in range(count):
            self.episode += 1
            task = (self.episode, max_steps, self.result_queue)
            try:
                self.task_queue.put(task, timeout=1.0)
            except queue.Full:
                break
    
    def collect_results(self, timeout=0.5) -> List[Dict]:
        """Collect completed episode results"""
        results = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                result = self.result_queue.get(timeout=0.1)
                results.append(result)
            except queue.Empty:
                continue
        
        return results
    
    def process_results(self, results: List[Dict]):
        """
        Process results and add transitions to replay buffer WITH METADATA.
        """
        for result in results:
            worker_id = result['worker_id']
            episode_num = result['episode_num']
            transitions = result['transitions']
            episode_reward = result['episode_reward']
            step_count = result['step_count']
            
            # Add transitions to replay buffer WITH metadata
            for transition in transitions:
                if len(transition) == 7:
                    # New format with metadata
                    obs, action, reward, next_obs, done, policy, metadata = transition
                    
                    # Compute TD error estimate (if model available)
                    td_error = 1.0
                    try:
                        with torch.no_grad():
                            _, value, _, _, _, _ = self.model.initial_inference(
                                obs.unsqueeze(0).to(DEVICE),
                                generation=metadata['generation'],
                                terraform_pct=metadata['terraform_pct']
                            )
                            _, next_value, _, _, _, _ = self.model.initial_inference(
                                next_obs.unsqueeze(0).to(DEVICE),
                                generation=metadata['generation'],
                                terraform_pct=metadata['terraform_pct']
                            )
                            td_error = abs(reward + 0.997 * next_value.item() - value.item())
                    except:
                        pass
                    
                    # Update metadata with TD error
                    metadata['td_error'] = td_error
                    
                    # Add to replay with full metadata
                    self.replay.add(
                        obs=obs,
                        action=action,
                        reward=reward,
                        next_obs=next_obs,
                        done=done,
                        policy=policy,
                        generation=metadata['generation'],
                        terraform_pct=metadata['terraform_pct'],
                        td_error=td_error,
                        aux_targets=metadata['aux_targets']
                    )
                else:
                    # Old format without metadata (backward compatibility)
                    obs, action, reward, next_obs, done, policy = transition
                    self.replay.add(obs, action, reward, next_obs, done, policy)
            
            # Update statistics
            self.worker_stats[worker_id]['episodes'] += 1
            self.worker_stats[worker_id]['steps'] += step_count
            self.total_transitions_collected += len(transitions)
            
            # Log metrics
            self.manager.log_metric(f"rollout/worker_{worker_id}/episode_reward", episode_reward)
            self.manager.log_metric(f"rollout/worker_{worker_id}/episode_length", step_count)
            self.manager.log_metric("rollout/episode_reward", episode_reward)
            self.manager.log_metric("rollout/transitions_per_episode", len(transitions))
            
            if episode_num % 10 == 0:
                print(f"Worker {worker_id} | Episode {episode_num} | "
                      f"Reward: {episode_reward:.2f} | Steps: {step_count} | "
                      f"Transitions: {len(transitions)}")
    
    def get_max_steps_for_episode(self, episode_num: int) -> int:
        """Curriculum learning: gradually increase episode length"""
        if episode_num < 15:
            return episode_num * 2
        elif episode_num < 1000:
            return 30
        elif episode_num < 3000:
            return 50
        elif episode_num < 10000:
            return 75
        else:
            return 100
    
    def rollout_loop(self):
        """Main rollout loop"""
        print("Starting parallel rollout loop with metadata collection")
        
        batch_size = self.num_workers * 2
        
        while not self.stop_event.is_set():
            if len(self.replay) > 500_000:
                time.sleep(1)
                continue
            
            max_steps = self.get_max_steps_for_episode(self.episode)
            self.submit_episodes(batch_size, max_steps)
            
            time.sleep(5.0)
            
            results = self.collect_results(timeout=0.5)
            if results:
                self.process_results(results)
            
            if self.episode % 50 == 0:
                self.log_statistics()
    
    def log_statistics(self):
        """Log overall rollout statistics"""
        total_episodes = sum(stats['episodes'] for stats in self.worker_stats.values())
        total_steps = sum(stats['steps'] for stats in self.worker_stats.values())
        
        self.manager.log_metric("rollout/total_episodes", total_episodes)
        self.manager.log_metric("rollout/total_steps", total_steps)
        self.manager.log_metric("rollout/total_transitions", self.total_transitions_collected)
        self.manager.log_metric("rollout/replay_size", len(self.replay))
        
        episodes_per_worker = [stats['episodes'] for stats in self.worker_stats.values()]
        self.manager.log_metric("rollout/worker_balance_std", np.std(episodes_per_worker))
        
        print(f"\n=== Rollout Statistics (Episode {self.episode}) ===")
        print(f"Total Episodes: {total_episodes}")
        print(f"Total Steps: {total_steps}")
        print(f"Total Transitions: {self.total_transitions_collected}")
        print(f"Replay Buffer Size: {len(self.replay)}")
        print(f"Worker Balance: {episodes_per_worker}")
        print("=" * 50 + "\n")
    
    def stop(self):
        """Stop all workers gracefully"""
        print("Stopping parallel rollout manager...")
        self.stop_event.set()
        
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        print("Parallel rollout manager stopped")
    
    def wait_for_buffer_filled(self, min_size: int = 512):
        """Wait until replay buffer has at least min_size transitions"""
        print(f"Filling replay buffer to {min_size} transitions...")
        
        while len(self.replay) < min_size and not self.stop_event.is_set():
            if self.task_queue.qsize() < self.num_workers:
                max_steps = self.get_max_steps_for_episode(self.episode)
                self.submit_episodes(self.num_workers, max_steps)
            
            time.sleep(2.0)
            results = self.collect_results(timeout=0.5)
            if results:
                self.process_results(results)
            
            if len(self.replay) % 100 == 0:
                print(f"Replay buffer: {len(self.replay)}/{min_size}")
        
        print(f"✓ Replay buffer filled: {len(self.replay)} transitions")


if __name__ == "__main__":
    print("Fixed Parallel Rollout Manager with Metadata Support")
    print("Ready to use with PrioritizedReplayBuffer")