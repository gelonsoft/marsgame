from experiment import ExperimentManager
from ppo import Agent
from env import parallel_env

manager = ExperimentManager("runs/experiment")

agent = Agent(...)
opponent = Agent(...)

for generation in range(50):
    print("Training generation", generation)

    # train PPO here

    manager.save(agent, f"gen_{generation}")

    # evaluate
    manager.record_match(f"gen_{generation}", "baseline", winner=f"gen_{generation}")

    print("Leaderboard:", manager.leaderboard())
