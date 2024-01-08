import Agent.Hyperparameters as hp

import Main as m

def run():
    
    env, state_shape, action_shape = m.make_env("human")
    
    ddqAgent = m.make_agent(env, state_shape, action_shape)

    m.main(env, hp.TRAINING_STEPS, m.scripted_policy, ddqAgent.record_episode, ddqAgent.process_and_save_training_data)
    
    
run()
