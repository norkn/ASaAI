import numpy as np
from npy_append_array import NpyAppendArray
from time import localtime, strftime

import Agent.Hyperparameters as hp

import Main as m

def set_env_seed(env, seed):
    np.random.seed(int(seed))
    env.np_random = np.random

def run(num_episodes):
    seed = 0
    
    env, state_shape, action_shape = m.make_env(None)
    set_env_seed(env, seed)
    
    ddqAgent = m.load_agent(env, state_shape, action_shape)

    total = 0
    
    def in_loop(s, a, r, n, d):
      nonlocal total, ddqAgent
      total += r

      ddqAgent.record_episode(s, a, r, n, d)

    def end_episode():
      nonlocal total, ddqAgent

      print(strftime("%H:%M:%S", localtime()), "RL episode ends. total: ", total)
      NpyAppendArray(hp.RL_RESULTS_PATH, delete_if_exists = True).append(np.array([total]))
      total = 0

      ddqAgent.save_episode()
      ddqAgent.reset_episode()
      
    rl_result = m.main(env, num_episodes, ddqAgent.get_action, in_loop, end_episode)
