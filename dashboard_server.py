import os
import json
import time
import pandas as pd
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from experiment_manager import ExperimentManager

RUNS_DIR = "runs"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/experiments")
def experiments():
    runs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
    return jsonify(runs)

@app.route("/api/leaderboard/<run>")
def leaderboard(run):
    path = os.path.join(RUNS_DIR, run, "elo.json")
    if not os.path.exists(path):
        return jsonify([])
    with open(path) as f:
        data = json.load(f)
    return jsonify(sorted(data.items(), key=lambda x: -x[1]))

@app.route("/api/metrics/<run>")
def metrics(run):
    tb_dir = os.path.join(RUNS_DIR, run, "tensorboard")
    metrics = []

    for root, dirs, files in os.walk(tb_dir):
        for f in files:
            if "events" in f:
                metrics.append(os.path.join(root, f))

    return jsonify(metrics)

def broadcast_leaderboard():
    while True:
        for run in os.listdir(RUNS_DIR):
            path = os.path.join(RUNS_DIR, run, "elo.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                socketio.emit("leaderboard", {"run": run, "data": data})
        time.sleep(10)

if __name__ == "__main__":
    import threading
    t = threading.Thread(target=broadcast_leaderboard, daemon=True)
    t.start()
    socketio.run(app, host="0.0.0.0", port=8000)
