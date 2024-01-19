import Agent.Hyperparameters as hp

import Main as m

def run():
    
    env, state_shape, action_shape = m.make_env("human")
    
    ddqAgent = m.make_agent(env, state_shape, action_shape)

    def end_episode():
        ddqAgent.process_episode_and_save_training_data
        ddqAgent.save_episode()

    m.main(env, hp.TRAINING_NUM_EPISODES, m.scripted_policy, ddqAgent.record_episode, end_episode)
    
    
run()
