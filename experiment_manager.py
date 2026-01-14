import os
import time
import torch
from elo import Elo
from torch.utils.tensorboard import SummaryWriter
import json

class ExperimentManager:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        self.elo = Elo()
        self.best_agents = []   # [(name, elo)]
        self.last_save = time.time()

        self.writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
        self.global_step = 0

    def save(self, model, name):
        path = os.path.join(self.run_dir, f"{name}.pt")
        torch.save(model.state_dict(), path)

    def load_latest(self, model):
        files = [f for f in os.listdir(self.run_dir) if f.endswith(".pt")]
        if not files:
            return None
        latest = max(files, key=lambda x: os.path.getmtime(os.path.join(self.run_dir, x)))
        model.load_state_dict(torch.load(os.path.join(self.run_dir, latest)))
        return latest

    def register(self, name, elo):
        self.best_agents.append((name, elo))
        self.best_agents = sorted(self.best_agents, key=lambda x: -x[1])[:5]

        # Save Elo table
        path = os.path.join(self.run_dir, "elo.json")
        with open(path, "w") as f:
            json.dump(self.elo.rating, f, indent=2)
            
        # Log leaderboard
        for i, (agent, rating) in enumerate(self.best_agents):
            self.writer.add_scalar(f"leaderboard/{agent}", rating, self.global_step)


    def autosave(self, model, name):
        if time.time() - self.last_save > 120:
            self.save(model, name)
            self.last_save = time.time()
    def log_metric(self, name, value):
        self.writer.add_scalar(name, value, self.global_step)

    def step(self):
        self.global_step += 1