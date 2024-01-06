import Agent.DoubleDeepQAgent as ddqa
import Agent.Hyperparameters as hp

import Main as m

def run():
    
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

    m.main(env, hp.TRAINING_STEPS, ddqAgent.get_action, m.nop, m.nop)
    
    
run()
