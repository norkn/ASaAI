import numpy as np
from npy_append_array import NpyAppendArray

import Agent.Hyperparameters as hp

import Main as m

PATH = 'Savefiles/scripted_results.npy'

def set_env_seed(env, seed):
  np.random.seed(int(seed))
  env.np_random = np.random

def run():
    env, state_shape, action_shape = m.make_env(None)
    set_env_seed(env, 0)
    
    ddqAgent = m.make_agent(env, state_shape, action_shape)

    total = 0

    def in_loop(s, a, r, n, d):
      nonlocal total, ddqAgent
      total += r

      ddqAgent.record_episode(s, a, r, n, d)

    def end_episode():
        nonlocal total, ddqAgent
        print("scripted episode ends. total: ", total)
        NpyAppendArray(PATH, delete_if_exists = False).append(np.array([total]))
        total = 0
        
        ddqAgent.save_episode()
        ddqAgent.reset_episode()

    scripted_result = m.main(env, hp.TRAINING_NUM_EPISODES, m.scripted_policy, in_loop, end_episode)
    
run()
