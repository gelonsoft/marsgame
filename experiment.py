import os
import torch
from elo import EloTracker

class ExperimentManager:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.elo = EloTracker()

    def save(self, agent, name):
        path = os.path.join(self.run_dir, f"{name}.pt")
        torch.save(agent.state_dict(), path)

    def load(self, agent, name):
        path = os.path.join(self.run_dir, f"{name}.pt")
        agent.load_state_dict(torch.load(path))

    def record_match(self, a, b, winner):
        if winner == a:
            self.elo.update(a, b, 1)
        else:
            self.elo.update(a, b, 0)

    def leaderboard(self):
        return sorted(self.elo.ratings.items(), key=lambda x: -x[1])
