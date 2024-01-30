from gym import Wrapper
from gymnasium.spaces import Box

import numpy as np

import Environment.ObservationProcessor as op

class WrappedEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        state = self.env.observation_space.sample()
        state = op.ObservationProcessor.get_state(state)
        state_shape = (len(state), )
        
        self.observation_space = Box(shape=state_shape, low=0, high=np.inf)

    def _transform_observation(self, obs):
        return op.ObservationProcessor.get_state(obs)

    def _transform_reward(self, reward, obs):
        return reward 

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        obs = self._transform_observation(obs)
        reward = self._transform_reward(reward, obs)

        return obs, reward, terminated, truncated, info

    def reset(self):
        obs, info = self.env.reset()

        obs = self._transform_observation(obs)

        return obs, info

