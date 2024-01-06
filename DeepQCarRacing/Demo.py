import Agent.Hyperparameters as hp

import Main as m

def run():
    
    env, state_shape, action_shape = m.make_env("human")
    
    ddqAgent = m.load_agent(env, state_shape, action_shape)

    m.main(env, hp.TRAINING_STEPS, ddqAgent.get_action, m.nop, m.nop)
    
    
run()
