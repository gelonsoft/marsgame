import glob
import os
import random
import string
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

#from pettingzoo.butterfly import pistonball_v6

from env import parallel_env

os.environ['CUDA_PATH']='e:/PC/cuda126'
continue_train=os.getenv('CONTINUE_TRAIN', 'False') == 'True'
model_path=os.getenv('MODEL_PATH', 'ppo_model.pt') 
run_name=os.getenv('RUN_NAME', '')
if run_name=="":
    run_name=''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
batch_size = int(os.getenv('BATCH_SIZE', "512"))
max_cycles = int(os.getenv('MAX_CYCLES', "1000"))
total_episodes = int(os.getenv('TOTAL_EPISODES', "1000"))
start_lr=float(os.getenv('START_LR', 0.001))
save_last_n=int(os.getenv('SAVE_LAST_N', 5))
save_interval=int(os.getenv('SAVE_INTERVAL', 5))

print("=== Parameters ====")
print(f"CONTINUE_TRAIN: {continue_train}")
print(f"MODEL_PATH: {model_path}")
print(f"RUN_NAME: {run_name}")
print(f"START_LR: {start_lr}")
print(f"BATCH_SIZE: {batch_size}")
print(f"MAX_CYCLES: {max_cycles}")
print(f"TOTAL_EPISODES: {total_episodes}")
print(f"SAVE_LAST_N: {save_last_n}")
print(f"SAVE_INTERVAL: {save_interval}")
print("====================")


writer = SummaryWriter(f"runs/{run_name}")

writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n|CONTINUE_TRAIN|{continue_train}|\n|MODEL_PATH|{model_path}|\n|RUN_NAME|{run_name}|\n|START_LR|{start_lr}|\n|BATCH_SIZE|{batch_size}|\n|MAX_CYCLES|{max_cycles}|\n|TOTAL_EPISODES|{total_episodes}|\n|SAVE_LAST_N|{save_last_n}|\n|SAVE_INTERVAL|{save_interval}|"
)


start_time = time.time()

def mask_logits_old(logits, action_mask):
    """
    Modify logits by masking rightmost elements based on action_mask.
    
    Args:
        logits: Input tensor of shape (batch_size, LOGITS_SIZE)
        action_mask: List of integers where each integer represents how many 
                    rightmost elements to mask in each row (0 means mask all)
    
    Returns:
        Modified logits tensor with appropriate elements masked
    """
    # Ensure inputs are on the same device
    device = logits.device
    batch_size, logits_size = logits.shape
    
    modified_logits=logits.clone()
    # Convert action_mask to a tensor
    action_mask = torch.tensor(action_mask, dtype=torch.long, device=device)
    
    # Create a mask tensor initialized with -inf (to effectively disable those logits)
    modified_logits = torch.full_like(logits, float('-inf'))
    
    # For each row, keep the first (logits_size - mask_value) elements unmasked
    for i in range(batch_size):
        mask_value = action_mask[i]
        if mask_value == logits_size:
            # Mask no elements
            modified_logits[i]=logits[i]
        elif mask_value == 0:
            # Mask all elements
            modified_logits[i,0]=0
        else:
            # Keep first (logits_size - mask_value) elements, mask the rest
            if mask_value > 0:
                modified_logits[i, :mask_value] = logits[i,:mask_value]
    
    # Apply the mask to the logits
    #modified_logits = logits + mask
    
    return modified_logits

def mask_logits(logits, valid_action_counts):
    """
    Masks invalid actions by setting logits to -inf.
    valid_action_counts: list[int] — number of valid actions per batch row
    """
    device = logits.device
    batch_size, num_actions = logits.shape

    masked = torch.full_like(logits, float('-inf')).to(device)

    for i, valid_n in enumerate(valid_action_counts):
        if valid_n > 0:
            masked[i, :valid_n] = logits[i, :valid_n]

    return masked

class Agent(nn.Module):
    def __init__(self, obs_size, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(obs_size, 4096)),
            nn.ReLU(),
            self._layer_init(nn.Linear(4096, 512)),
            nn.ReLU(),
            self._layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
        )

        self.actor = self._layer_init(nn.Linear(256, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(256, 1), std=1.0)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None, action_mask=None):
        hidden = self.network(x)
        logits = mask_logits(self.actor(hidden), action_mask)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    
    obs = np.stack([obs[a] for a in obs], axis=0)
    #print(obs.shape)
    # transpose to be (batch, channel, height, width)
    #obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: int(x[i]) for i, a in enumerate(env.possible_agents)}

    return x


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.01
    vf_coef = 0.5
    clip_coef = 0.2
    gamma = 0.99
    start_lr = 3e-4
    gae_lambda = 0.95


    

    """ ENV SETUP """
    env = parallel_env()
    #env = color_reduction_v0(env)
    #env = resize_v1(env, frame_size[0], frame_size[1])
    #env = frame_stack_v1(env, stack_size=stack_size)
    num_agents = len(env.possible_agents)
    #num_actions = env.action_space(env.possible_agents[0]).shape[0]
    num_actions = env.action_space(env.possible_agents[0]).shape[0]
    observation_size = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """
    print(f"Num actions is {num_actions} space={num_actions}")
    agent = Agent(obs_size=observation_size[0],num_actions=num_actions)
    print("NN created")
    optimizer = optim.Adam(agent.parameters(), lr=start_lr, eps=1e-5)
    print("Optimizer created")
    if continue_train:
        checkpoint=torch.load(model_path)
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded model from {model_path}")

    agent.to(device)
    print("NN sent to device")
    
    #scheduler = StepLR(optimizer, step_size=30,gamma=0.1)
    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, observation_size[0] )).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)
    rb_action_masks=torch.zeros((max_cycles, num_agents)).to(device)
    print("Training started")
    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=None)
            
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                obs = batchify_obs(next_obs, device)

                # get action from the agent
                action_mask=env.get_action_mask()
                actions, logprobs, _, values = agent.get_action_and_value(obs, action_mask=action_mask)
                for ii in range(len(env.agents)):
                    rb_action_masks[step][ii]=action_mask[ii]
                #print(f"action_mask={action_mask} rb_action_masks.shape={rb_action_masks.shape} rb_action_masks={rb_action_masks.cpu()}")
                # execute the environment and log data
                

                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )
                for agent_id in env.agents:
                    writer.add_scalar(f"charts/train-player{agent_id}", rewards[agent_id], episode)
                
                #print(f"Next obs: {next_obs}")

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                end_step = step
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    print(f"Termination or truncation reached: {terms}, {truncs}")
                    
                    break
                
        print(f"End step: {end_step}")

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * (1 - rb_terms[t + 1])
                    - rb_values[t]
                )

                rb_advantages[t] = delta + gamma * gae_lambda * rb_advantages[t + 1] * (1 - rb_terms[t + 1])
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)
        b_action_masks=torch.flatten(rb_action_masks[:end_step], start_dim=0, end_dim=1)
        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            print(f"Repeat {repeat} of 3: {0, len(b_obs), batch_size} end_step={end_step}")
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]
                #print(f"Batch index: {batch_index} action_mask={b_action_masks.long()[batch_index].cpu()}")

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index],
                    b_actions.long()[batch_index],
                    action_mask=b_action_masks.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantages
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        print("===")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var.item()}")
        print("\n-------------------------------------------\n")
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], episode)
        writer.add_scalar("losses/value_loss", v_loss.item(), episode)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), episode)
        writer.add_scalar("losses/entropy", entropy_loss.item(), episode)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), episode)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), episode)
        writer.add_scalar("losses/explained_variance", explained_var, episode)
        print("SPS:", int(episode / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(episode / (time.time() - start_time)), episode)
        
        if episode % save_interval == 0: 
            torch.save({"model_state_dict":agent.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, f"runs/{run_name}/model_{episode}.pt")
            print(f"Model saved at runs/{run_name}/model_{episode}.pt")
            model_files=glob.glob(f"runs/{run_name}/model_*.pt")
            if len(model_files) > save_last_n:
                model_files.sort(key=os.path.getmtime)
                os.remove(model_files[0])           
                print(f"Removed oldest model file {model_files[0]} to keep {save_last_n} models")
        #scheduler.step()
                      
    """ RENDER THE POLICY """
    env = parallel_env()
    #env = color_reduction_v0(env)
    #env = resize_v1(env, 64, 64)
    #env = frame_stack_v1(env, stack_size=4)

    agent.eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs, infos = env.reset(seed=None)
            obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            for cycle in range(max_cycles):
                action_mask=env.get_action_mask()
                actions, logprobs, _, values = agent.get_action_and_value(obs,action_mask=action_mask)
                #print(f"Agent eval step: actions={actions}")
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                for agent_id in env.agents:
                    writer.add_scalar(f"charts/eval-player{agent_id}", episode, rewards[agent_id])
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]
                if any(terms) and any(truncs):
                    print("Termination or truncation detected. Ending episode.")
                    break
    
    
    

    

writer.close()