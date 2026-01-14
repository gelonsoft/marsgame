from trainer import BackgroundTrainer
from env import parallel_env
from observe_gamestate import observe

env = parallel_env()
obs, _ = env.reset()

obs_dim = len(obs["1"])
action_dim = env.action_space("1").n

trainer = BackgroundTrainer(obs_dim, action_dim)
trainer.start()
