import torch
import rl_acid_environment
import environment_wrapper


config = {"horizon": 3}
config["obs_dim"] = 4*config["horizon"]
env = environment_wrapper.GenerateEnvironmentWrapper(env_name, config)

