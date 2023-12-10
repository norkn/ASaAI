from gym import Wrapper
from gymnasium.spaces import Box

import numpy as np

import ObservationProcessor as op

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
        reward -= obs[8] #being on grass gets punished more
        reward -= int(obs[0] < 2) #being slow gets punished
        reward += int(obs[3] == 0 and obs[2] > 1) #turning right gets incentivized when near vision left is grass
        reward += int(obs[4] == 0 and obs[1] > 1) #turning left gets incentivized when near vision right is grass

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

