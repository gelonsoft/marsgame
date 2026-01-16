from trainer import BackgroundTrainer
from env_all_actions import parallel_env
from observe_gamestate import observe
from myconfig import MAX_ACTIONS

env = parallel_env()
obs, infos, action_count, action_list = env.reset()

obs_dim = len(obs)
action_dim = MAX_ACTIONS

trainer = BackgroundTrainer(obs_dim, action_dim)
trainer.start()
