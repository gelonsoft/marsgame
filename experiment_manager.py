import os
import time
import torch
from elo import Elo
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime



class ExperimentManager:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        self.elo = Elo()
        self.elo_path = os.path.join(run_dir, "elo.json")
        self.best_agents = []   # [(name, elo)]
        self.last_save = time.time()
        now = datetime.now()
        format_string = "%Y-%m-%d-%H-%M-%S"
        formatted_datetime = now.strftime(format_string)
        self.writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard",formatted_datetime))
        self.global_step = 0

        # Load previous Elo if exists
        if os.path.exists(self.elo_path):
            with open(self.elo_path) as f:
                self.elo.rating = json.load(f)

        self.best_pool = []   # list of (name, model)
        self.agent_id = 0

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
        self.writer.add_scalar("elo/" + name, elo, self.global_step)

    def autosave(self, model, name):
        if time.time() - self.last_save > 120:
            self.save(model, name)
            self.last_save = time.time()
    def log_metric(self, name, value):
        self.writer.add_scalar(name, value, self.global_step)

    def step(self):
        self.global_step += 1