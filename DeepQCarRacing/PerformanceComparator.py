import numpy as np
from npy_append_array import NpyAppendArray
import time

import Agent.Hyperparameters as hp

import Main as m

def set_env_seed(env, seed):
  np.random.seed(int(seed))
  env.np_random = np.random

def run():
    seed = int(time.time())
    total = 0

    def in_loop(s, a, r, n, d):
      nonlocal total, ddqAgent
      total += r

      ddqAgent.record_episode(s, a, r, n, d)

    def end_episode():
      nonlocal total, ddqAgent
      print("episode ends. total: ", total)
      total = 0

      ddqAgent.save_episode()
      ddqAgent._reset_episode()
      
    ###################
    print("RL results:")
    env, state_shape, action_shape = m.make_env(None)
    set_env_seed(env, seed)
    
    ddqAgent = m.load_agent(env, state_shape, action_shape)


    rl_result = m.main(env, hp.RUNNING_NUM_EPISODES, ddqAgent.get_action, in_loop, end_episode)
    NpyAppendArray('Savefiles/rl_results.npy',   delete_if_exists = False).append(np.array([rl_result]))
    
    #########################
    print("Scripted results:")
    env, state_shape, action_shape = m.make_env(None)
    set_env_seed(env, seed)


    env, _, _ = m.make_env(None)

    scripted_result = m.main(env, hp.RUNNING_NUM_EPISODES, m.scripted_policy, in_loop, end_episode)
    NpyAppendArray('Savefiles/scripted_results.npy',   delete_if_exists = False).append(np.array([scripted_result]))
    
#run()
