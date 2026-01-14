import os
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = "runs"

def load_elo(run):
    with open(os.path.join(RUNS_DIR, run, "elo.json")) as f:
        return json.load(f)

def compare():
    runs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
    tables = []

    for run in runs:
        try:
            elo = load_elo(run)
            for agent, rating in elo.items():
                tables.append({"run": run, "agent": agent, "elo": rating})
        except:
            pass

    df = pd.DataFrame(tables)
    print(df)

    for run in df["run"].unique():
        sub = df[df["run"] == run]
        plt.plot(sub["agent"], sub["elo"], label=run)

    plt.legend()
    plt.title("Experiment Comparison")
    plt.ylabel("Elo")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    compare()
