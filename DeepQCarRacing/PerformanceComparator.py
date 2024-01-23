import numpy as np
from npy_append_array import NpyAppendArray
import time

import Agent.Hyperparameters as hp

import Main as m

PATH = 'Savefiles/rl_results.npy'

def set_env_seed(env, seed):
  np.random.seed(int(seed))
  env.np_random = np.random

def run():
    seed = 0#int(time.time())
    
    env, state_shape, action_shape = m.make_env("human")#None)
    set_env_seed(env, seed)
    
    ddqAgent = m.load_agent(env, state_shape, action_shape)

    total = 0
    
    def in_loop(s, a, r, n, d):
      nonlocal total, ddqAgent
      total += r

      ddqAgent.record_episode(s, a, r, n, d)

    def end_episode():
      nonlocal total, ddqAgent
      print("RL episode ends. total: ", total)
      NpyAppendArray(PATH, delete_if_exists = False).append(np.array([total]))
      total = 0

      ddqAgent.save_episode()
      ddqAgent.reset_episode()
      
    rl_result = m.main(env, hp.TRAINING_NUM_EPISODES, ddqAgent.get_action, in_loop, end_episode)
    
run()
