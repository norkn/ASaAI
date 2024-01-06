import numpy as np
from npy_append_array import NpyAppendArray

import Agent.DoubleDeepQAgent as ddqa
import Agent.Hyperparameters as hp

import Main as m

def run():
    np.random.seed(0)
    env, state_shape, action_shape = m.make_env("human")
    
    ddqAgent = ddqa.DoubleDeepQAgent.load(env,
                                          state_shape,
                                          action_shape,
                                          hp.FILENAME,
                                          hp.NUM_BATCHES,
                                          hp.EPOCHS,
                                          hp.SAMPLE_SIZE, 
                                          hp.GAMMA, 
                                          hp.EPSILON_DECAY)


    rl_result = m.main(env, hp.RUNNING_STEPS, ddqAgent.get_action, m.nop, m.nop)
    NpyAppendArray('Savefiles/rl_results.npy',   delete_if_exists = False).append(np.array([rl_result]))
    
    #########################
    np.random.seed(0)
    env, _, _ = m.make_env("human")

    scripted_result = m.main(env, hp.RUNNING_STEPS, m.scripted_policy, m.nop, m.nop)
    NpyAppendArray('Savefiles/scripted_results.npy',   delete_if_exists = False).append(np.array([scripted_result]))
    
run()
