import time
import os
import gym
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import robosuite as suite
from robosuite.wrappers import GymWrapper

from networks import CriticNetwork, ActorNetwork
from buffer import ReplayBuffer

if __name__ == '__main__':

    # Create directories safely
    os.makedirs("tmp/td3", exist_ok=True)

    env_name = "Door"

    env = suite.make(
        env_name=env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(
            default_controller="JOINT_VELOCITY"
        ),
        use_camera_obs = False,
        has_renderer=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)


    ###
    #critic_network = CriticNetwork([8], 8)
    #actor_network = ActorNetwork([8], 8)
    
    # replay_buffer = ReplayBuffer(8, [8], 8)
