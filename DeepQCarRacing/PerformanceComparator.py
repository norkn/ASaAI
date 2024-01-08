import numpy as np
from npy_append_array import NpyAppendArray

import Agent.Hyperparameters as hp

import Main as m

def run():
    np.random.seed(0)
    env, state_shape, action_shape = m.make_env(None)
    
    ddqAgent = m.load_agent(env, state_shape, action_shape)


    rl_result = m.main(env, hp.RUNNING_NUM_EPISODES, ddqAgent.get_action, m.nop, m.nop)
    NpyAppendArray('Savefiles/rl_results.npy',   delete_if_exists = False).append(np.array([rl_result]))
    
    #########################
    np.random.seed(0)
    env, _, _ = m.make_env(None)

    scripted_result = m.main(env, hp.RUNNING_NUM_EPISODES, m.scripted_policy, m.nop, m.nop)
    NpyAppendArray('Savefiles/scripted_results.npy',   delete_if_exists = False).append(np.array([scripted_result]))
    
run()
