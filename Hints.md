Training an AI bot for a complex game like **Terraforming Mars** using libraries like **`gymnasium`**, **`pettingzoo`**, and **`torch`** involves several steps, especially if you want the agent to automatically learn **important parameters** (i.e., features) instead of manually hand-coding reward weights.

---

## 🧠 Objective

You want to train an agent that:

* Learns which **game state parameters** (features) are important,
* Optimizes its actions to **maximize victory / score**,
* Learns using **reinforcement learning (RL)**.

---

## 🧩 Key Concepts

| Component              | Purpose                                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| `gymnasium`            | Defines the environment interface: `reset`, `step`, `observation`, etc.  |
| `pettingzoo`           | Multi-agent environment wrapper around `gymnasium`                       |
| `torch`                | Deep learning (used to build the agent’s neural networks)                |
| **Reward function**    | Your current code (can be used, but ideally the model learns on its own) |
| **Feature extraction** | Model automatically learns which fields from state are useful            |

---

## ✅ Steps to Train the Bot

---

### 1. **Define the Environment (Wrap Terraforming Mars as PettingZoo Environment)**

This is the hardest and most custom part:

```python
from pettingzoo import AECEnv
from gymnasium.spaces import Dict, Discrete, Box

class TerraformingMarsEnv(AECEnv):
    metadata = {"render_modes": []}

    def __init__(self):
        self.possible_agents = ["player_0", "player_1", "player_2"]
        self.action_spaces = {agent: Discrete(N_ACTIONS) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Dict({
                "game_state": Box(low=0, high=1e6, shape=(N_FEATURES,), dtype=float)
            }) for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        # Load or initialize game_state
        pass

    def step(self, action):
        # Apply action to game state, update rewards and turn
        pass

    def observe(self, agent):
        # Return processed vector of features (this is where auto-parameter learning happens)
        pass
```

**Tips**:

* You’ll need to encode your `PlayerViewModel` and `GameModel` JSON into flat vectors (`Box`) or dicts (`Dict`) for each agent.
* Normalize numerical values like `terraformRating` or `plants`.
* Use autoencoders or embeddings for categorical features like tags or cards if you want to learn them.

---

### 2. **Build a PyTorch Agent**

```python
import torch
import torch.nn as nn

class MarsAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)
```

* `input_dim` = size of feature vector extracted from game state
* `output_dim` = number of legal actions

---

### 3. **Train the Agent with RL (PPO, A2C, etc.)**

Use a library like `stable-baselines3` (for single-agent) or `supersuit` + `pettingzoo` + `RLlib` for multi-agent:

```python
from stable_baselines3 import PPO
from pettingzoo.utils.conversions import parallel_wrapper_fn
from supersuit import pettingzoo_env_to_vec_env_v1

parallel_env = parallel_wrapper_fn(TerraformingMarsEnv)()
vec_env = pettingzoo_env_to_vec_env_v1(parallel_env)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000000)
```

---

### 4. **Let the Agent Discover Important Parameters**

* Don’t provide reward weights manually — the agent receives raw state features and learns from **victory points / game outcomes**.
* You can monitor the learned weights by examining the **first-layer weights** in the `MarsAgent` neural network. These implicitly indicate which features are important.

---

## ✅ Bonus: If You Want Explainability

To extract which parameters are influential, use **feature attribution techniques**:

```python
import shap
explainer = shap.Explainer(agent.net, feature_names=feature_names)
shap_values = explainer(state_batch)
shap.plots.beeswarm(shap_values)
```

This shows which game features (like `terraformRating`, `plants`, etc.) are driving decisions.

---

## 🔚 Summary

| What You Want                     | How To Do It                                                              |
| --------------------------------- | ------------------------------------------------------------------------- |
| Train AI to play TM               | Wrap TM logic in `PettingZoo` + `Gymnasium` environment                   |
| Learn which features matter       | Feed all features as inputs, let NN weights learn importance (via reward) |
| Avoid handcrafting reward weights | Let `reward = game_score` or `VP`, and let agent optimize that            |
| Extract feature importance        | Use SHAP or analyze neural net weights                                    |

---

Would you like a **starter repo or notebook scaffold** for wrapping Terraforming Mars into a PettingZoo environment?
