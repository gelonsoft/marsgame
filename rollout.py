"""
Parallel Rollout Implementation for MuZero Training
Runs 6 concurrent rollout threads to fill replay buffer faster

FACTORED ACTION INTEGRATION:
----------------------------
This module has been updated to work with the factored (hierarchical) action
architecture. Key changes:

1. MCTS.run() now receives factored action metadata from env.step():
   - head_masks: per-head legal action masks (4 arrays)
   - cond_masks_*: conditional masks for hierarchical selection
   - factored_legal: dict mapping slots to (type, obj, pay, extra) tuples

2. Observation space is now 512-dim (pure game state, no encoded actions).

3. env.step() returns 10-tuple instead of 7-tuple, including factored data.

4. Workers cache factored data between steps to pass to next MCTS call.

5. legal_actions_decoder.get_legal_actions_from_obs is no longer used
   (legal actions come directly from env.step()).
"""

import copy
import random
import torch
import threading
import time
import queue
import numpy as np
from typing import List, Dict, Tuple
from env_all_actions import parallel_env
from mcts import MCTS
from factored_actions import FactoredActionEncoder, FACTORED_ACTION_DIMS, ACTION_TYPE_DIM, OBJECT_ID_DIM, PAYMENT_ID_DIM, EXTRA_PARAM_DIM, NUM_HEADS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RolloutWorker:
    """
    Individual worker thread that performs rollouts.
    Each worker has its own environment and MCTS instance.
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
        
        # Each worker has its own environment
        self.env = parallel_env()
        
        # Each worker has its own MCTS (not shared due to tree state)
        self.mcts = MCTS(
            model=model,
            action_dim=action_dim,
            **mcts_config
        )
        
        self.episodes_completed = 0
        self.total_steps = 0
        self.factored_encoder = FactoredActionEncoder()
        
    def run_episode(self, episode_num: int, max_steps: int) -> List[Tuple]:
        """
        Run a single episode and return list of transitions.
        Returns: List of (obs, action, reward, next_obs, done, policy)
        """
        transitions = []
        replay_data={}
        try:
            obs, infos, action_count, action_list, current_env_id = self.env.reset()
            
            # Initialize factored action data attributes (will be set after first step)
            self.last_head_masks = None
            self.last_cond_obj = None
            self.last_cond_pay = None
            self.last_cond_extra = None
            self.last_factored_legal = None
            
            episode_reward = 0
            step_count = 0
            done = False
            
            # Dynamic temperature based on episode number
            if episode_num < 2000:
                base_temperature = 1.5
            elif episode_num < 10000:
                base_temperature = 1.0
            else:
                base_temperature = max(0.25, 1.0 - (episode_num - 10000) / 40000)
            
            with torch.no_grad():
                while not done and step_count < max_steps:
                    obs_vec = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                    
                    # Dynamic temperature annealing within episode
                    temperature = max(0.5, base_temperature - (step_count / max_steps) * 0.5)
                    
                    # Run MCTS to get policy
                    # Note: head_masks and conditional masks come from previous env.step()
                    # On first step they'll be None and MCTS will use safe defaults
                    policy = torch.tensor(
                        self.mcts.run(
                            obs_vec,
                            legal_actions=action_list,
                            temperature=temperature,
                            training=True,
                            head_masks=getattr(self, 'last_head_masks', None),
                            cond_masks_obj_by_type=getattr(self, 'last_cond_obj', None),
                            cond_masks_pay_by_obj=getattr(self, 'last_cond_pay', None),
                            cond_masks_extra_by_pay=getattr(self, 'last_cond_extra', None),
                            factored_legal=getattr(self, 'last_factored_legal', None)
                        )
                    ).detach().cpu()
                    
                    # Normalize policy
                    policy = policy + 1e-8
                    policy = policy / policy.sum()
                    
                    # Select action
                    action = 0
                    if action_count > 0:
                        # Use legal actions directly from env (already available in action_list)
                        legal_actions = action_list
                        
                        # Apply mask to policy
                        mask = np.zeros(self.action_dim)
                        mask[legal_actions] = 1.0
                        
                        masked_policy = policy.numpy() * mask
                        masked_policy_sum = masked_policy.sum()
                        
                        if masked_policy_sum > 1e-18:
                            masked_policy = masked_policy / masked_policy_sum
                        else:
                            masked_policy = mask / len(legal_actions)
                        
                        masked_policy2 = np.clip(masked_policy, 1e-18, 1e6)
                        
                        try:
                            action = int(np.random.choice(self.action_dim, p=masked_policy2))
                        except:
                            action = random.choice(legal_actions)
                        
                        if action not in legal_actions:
                            action = random.choice(legal_actions)
                    
                    # Take action in environment
                    # env.step() now returns 10-tuple with factored action data
                    next_obs, rewards, terms, action_count, action_list, all_rewards, curr_eid, head_masks, cond_obj, cond_pay, cond_extra, factored_legal = self.env.step(action)
                    
                    # Store factored action data for next MCTS call
                    self.last_head_masks = head_masks
                    self.last_cond_obj = cond_obj
                    self.last_cond_pay = cond_pay
                    self.last_cond_extra = cond_extra
                    self.last_factored_legal = factored_legal
                    
                    # Clip reward
                    reward = float(max(-2.0, min(2.0, rewards)))
                    episode_reward += reward
                    done = terms
                    step_count += 1
                    
                    next_vec = torch.tensor(next_obs, dtype=torch.float32)
                    
                    if curr_eid == current_env_id:
                        transitions.append((
                            obs_vec.cpu(),
                            action,
                            reward,
                            next_vec,
                            done,
                            policy
                        ))
                    else:
                        replay_data[current_env_id] = {
                            'obs': obs_vec.cpu(),
                            'action': action,
                            'reward': reward,
                            'done': done,
                            'policy': policy
                        }
                        if curr_eid in replay_data:
                            r = replay_data[curr_eid]
                            transitions.append((r['obs'], r['action'], r['reward'], next_vec, r['done'], r['policy']))
                        current_env_id = curr_eid
                    
                    
                    obs = next_obs
            
            self.episodes_completed += 1
            self.total_steps += step_count
            
            return transitions, episode_reward, step_count
            
        except Exception as e:
            print(f"Worker {self.worker_id} error in episode: {e}")
            import traceback
            traceback.print_exc()
            return [], 0.0, 0
    
    def worker_loop(self):
        """Main loop for worker thread"""
        print(f"Worker {self.worker_id} started")
        
        while not self.stop_event.is_set():
            try:
                # Get task from queue (episode_num, max_steps)
                task = self.worker_queue.get(timeout=1.0)
                
                if task is None:  # Poison pill
                    break
                
                episode_num, max_steps = task
                
                # Run episode
                transitions, episode_reward, step_count = self.run_episode(episode_num, max_steps)
                
                # Return results via queue
                result = {
                    'worker_id': self.worker_id,
                    'episode_num': episode_num,
                    'transitions': transitions,
                    'episode_reward': episode_reward,
                    'step_count': step_count
                }
                print("Worker loop...")
                
                # Put result back (will be consumed by main thread)
                self.worker_queue.task_done()
                
                # Yield result through a separate result queue
                # (This will be set up in ParallelRolloutManager)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Worker {self.worker_id} stopped")


class ParallelRolloutManager:
    """
    Manages multiple rollout workers running in parallel.
    Coordinates task distribution and result collection.
    """
    
    def __init__(
        self,
        model,
        replay_buffer,
        manager,
        num_workers: int = 6,
        action_dim: int = 64,
        obs_dim: int = 512,  # Updated: no more encoded actions in obs
        mcts_config: dict = None
    ):
        self.model = model
        self.replay = replay_buffer
        self.manager = manager
        self.num_workers = num_workers
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        
        # Default MCTS config
        if mcts_config is None:
            mcts_config = {
                'sims': 100,
                'c_puct': 1.5,
                'discount': 0.997,
                'dirichlet_alpha': 0.3,
                'dirichlet_frac': 0.25
            }
        self.mcts_config = mcts_config
        
        # Threading infrastructure
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Create workers
        self.workers: List[RolloutWorker] = []
        self.worker_threads: List[threading.Thread] = []
        
        for i in range(num_workers):
            worker = RolloutWorker(
                worker_id=i,
                model=model,
                action_dim=action_dim,
                obs_dim=obs_dim,
                mcts_config=mcts_config,
                worker_queue=self.task_queue,
                lock=self.lock,
                stop_event=self.stop_event
            )
            self.workers.append(worker)
        
        # Episode counter
        self.episode = 0
        self.total_transitions_collected = 0
        
        # Statistics
        self.worker_stats = {i: {'episodes': 0, 'steps': 0} for i in range(num_workers)}
        
    def start(self):
        """Start all worker threads"""
        for i, worker in enumerate(self.workers):
            thread = threading.Thread(
                target=self._worker_wrapper,
                args=(worker,),
                daemon=True,
                name=f"RolloutWorker-{i}"
            )
            thread.start()
            self.worker_threads.append(thread)
        
        print(f"Started {self.num_workers} rollout workers")
    
    def _worker_wrapper(self, worker: RolloutWorker):
        """
        Wrapper for worker loop that handles result collection.
        Each worker runs episodes and puts results in result_queue.
        """
        while not self.stop_event.is_set():
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:  # Poison pill
                    break
                
                episode_num, max_steps = task
                
                # Run episode
                transitions, episode_reward, step_count = worker.run_episode(episode_num, max_steps)
                
                # Put result in result queue
                result = {
                    'worker_id': worker.worker_id,
                    'episode_num': episode_num,
                    'transitions': transitions,
                    'episode_reward': episode_reward,
                    'step_count': step_count
                }
                self.result_queue.put(result)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker.worker_id} wrapper error: {e}")
                import traceback
                traceback.print_exc()
    
    def submit_episodes(self, num_episodes: int, max_steps: int):
        """
        Submit multiple episodes to the task queue.
        Workers will pick them up and process them in parallel.
        """
        for _ in range(num_episodes):
            self.episode += 1
            self.task_queue.put((self.episode, max_steps))
    
    def collect_results(self, timeout: float = 1.0) -> List[Dict]:
        """
        Collect completed episode results from result queue.
        Returns list of result dictionaries.
        """
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
        Process results and add transitions to replay buffer.
        """
        for result in results:
            worker_id = result['worker_id']
            episode_num = result['episode_num']
            transitions = result['transitions']
            episode_reward = result['episode_reward']
            step_count = result['step_count']
            
            # Add transitions to replay buffer
            for obs, action, reward, next_obs, done, policy in transitions:
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
        """
        Curriculum learning: gradually increase episode length.
        """
        if episode_num < 15:
            return episode_num*2
        elif episode_num < 1000:
            return 30
        elif episode_num < 3000:
            return 50
        elif episode_num < 10000:
            return 75
        else:
            return 100
    
    def rollout_loop(self):
        """
        Main rollout loop that coordinates workers.
        This replaces the single-threaded rollout_loop in trainer.py.
        """
        print("Starting parallel rollout loop")
        
        # Keep workers busy
        batch_size = self.num_workers * 2  # 2 episodes per worker in queue
        
        while not self.stop_event.is_set():
            # Check replay buffer size
            if len(self.replay) > 500_000:
                time.sleep(1)
                continue
            
            # Determine max_steps based on current episode number
            max_steps = self.get_max_steps_for_episode(self.episode)
            
            # Submit batch of episodes
            self.submit_episodes(batch_size, max_steps)
            
            # Wait for some results (collect every 5 seconds)
            time.sleep(5.0)
            
            # Collect and process results
            results = self.collect_results(timeout=0.5)
            if results:
                self.process_results(results)
            
            # Log overall statistics every 50 episodes
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
        
        # Worker balance
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
        
        # Send poison pills
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        print("Parallel rollout manager stopped")
    
    def wait_for_buffer_filled(self, min_size: int = 512):
        """
        Wait until replay buffer has at least min_size transitions.
        Useful for initial buffer filling before training starts.
        """
        print(f"Filling replay buffer to {min_size} transitions...")
        
        while len(self.replay) < min_size and not self.stop_event.is_set():
            # Submit episodes if queue is getting empty
            if self.task_queue.qsize() < self.num_workers:
                max_steps = self.get_max_steps_for_episode(self.episode)
                self.submit_episodes(self.num_workers, max_steps)
            
            # Collect results
            time.sleep(2.0)
            results = self.collect_results(timeout=0.5)
            if results:
                self.process_results(results)
            
            # Progress update
            if len(self.replay) % 100 == 0:
                print(f"Replay buffer: {len(self.replay)}/{min_size}")
        
        print(f"Replay buffer filled: {len(self.replay)} transitions")



if __name__ == "__main__":
    """
    Standalone test of parallel rollout system
    """
    print("Testing Parallel Rollout Manager...")
    
    # Mock objects for testing
    from muzero import MuZeroNet
    from replay import ReplayBuffer
    
    class MockManager:
        def log_metric(self, name, value):
            pass
    
    # Create model and replay buffer
    model = MuZeroNet(obs_dim=512, action_dim=64).to(DEVICE)
    replay = ReplayBuffer(
        capacity=10000,
        obs_dim=512,
        action_dim=64,
        device=DEVICE,
        directory="test_replay",
        cache_size=1024
    )
    manager = MockManager()
    
    # Create parallel rollout manager
    parallel_rollout = ParallelRolloutManager(
        model=model,
        replay_buffer=replay,
        manager=manager,
        num_workers=6,
        action_dim=64,
        obs_dim=512  # Updated: no more encoded actions in obs
    )
    
    # Start workers
    parallel_rollout.start()
    
    # Fill buffer with 1000 transitions
    parallel_rollout.wait_for_buffer_filled(min_size=128)
    
    # Run for a bit
    print("\nRunning rollouts for 30 seconds...")
    time.sleep(30)
    
    # Stop
    parallel_rollout.stop()
    
    print(f"\nFinal replay buffer size: {len(replay)}")
    print("Test complete!")