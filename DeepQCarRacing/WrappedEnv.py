import gymnasium as gym
from gym import ObservationWrapper, RewardWrapper
from gymnasium.spaces import Box

import numpy as np

import ObservationProcessor as op

class WrappedEnv(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        state = env.observation_space.sample()
        state = op.ObservationProcessor.get_state(state)
        state_shape = (len(state), )
        
        self.observation_space = Box(shape=state_shape, low=0, high=np.inf)

    def observation(self, obs):
        return op.ObservationProcessor.get_state(obs)
