from trainer import BackgroundTrainer
from env_all_actions import parallel_env
from observe_gamestate import observe
from myconfig import MAX_ACTIONS
import argparse

parser = argparse.ArgumentParser(description="A script that reads two arguments.")

parser.add_argument("-t", "--disable-train", action="store_true", help="A boolean flag to disable trainer")
parser.add_argument("-p", "--enable-prom", action="store_true", help="A boolean flag to enable promotion")

args = parser.parse_args()


env = parallel_env()
obs, infos, action_count, action_list,current_env_id = env.reset()

obs_dim = len(obs)
action_dim = MAX_ACTIONS

print(obs_dim)
print(MAX_ACTIONS)

env=None
trainer = BackgroundTrainer(args,obs_dim, action_dim)
trainer.start()
