import torch
from ppo import Agent, batchify_obs, unbatchify
from env import parallel_env
from elo import EloTracker

def evaluate(model_a, model_b, games=20):
    elo = EloTracker()

    for g in range(games):
        env = parallel_env()
        obs, _ = env.reset()

        done = False
        while not done:
            obs_t = batchify_obs(obs, "cpu")
            a_action, _, _, _ = model_a.get_action_and_value(obs_t)
            obs, rewards, terms, truncs, infos = env.step(unbatchify(a_action, env))
            done = any(terms.values())

        winner = "A" if rewards["1"] > rewards["2"] else "B"
        elo.update("A", "B", 1 if winner == "A" else 0)

    print("Elo:", elo.ratings)
