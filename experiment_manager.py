import os
import time
import torch
from elo import Elo
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
import random


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

        self.best_pool_dir = os.path.join(run_dir, "best_pool")
        os.makedirs(self.best_pool_dir, exist_ok=True)

        self.promotion_stats_path = os.path.join(run_dir, "promotion_stats.json")

        if not os.path.exists(self.promotion_stats_path):
            with open(self.promotion_stats_path, "w") as f:
                json.dump([], f)

    def list_best_pool(self):
        return [os.path.join(self.best_pool_dir, f) for f in os.listdir(self.best_pool_dir) if f.endswith(".pt")]

    def list_last_agents(self):
        files= [
            os.path.join(self.run_dir, f)
            for f in os.listdir(self.run_dir)
            if f.endswith(".pt") and not f.startswith("best_pool")
        ]
        files.sort(key=os.path.getmtime,reverse=True)
        return files

    def save_promotion_stats(self, stats):
        try:
            with open(self.promotion_stats_path, "r") as f:
                data = json.load(f)
        except:
            data=[]
            pass
        data.append(stats)

        with open(self.promotion_stats_path, "w") as f:
            json.dump(data, f, indent=2)

    def save(self, model, name):
        path = os.path.join(self.run_dir, f"{name}.pt")
        torch.save(model.state_dict(), path)

    def load_latest(self, model):
        files = [os.path.join(self.run_dir, f) for f in os.listdir(self.run_dir) if f.endswith(".pt")]
        if not files:
            return None
        files.sort(key=os.path.getmtime,reverse=True)
        latest = files.pop(0)
        try:
            model.load_state_dict(torch.load(latest))
        except:
            latest = files.pop(0)
            model.load_state_dict(torch.load(latest))
        print(f"Loaded model for train {latest}")
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