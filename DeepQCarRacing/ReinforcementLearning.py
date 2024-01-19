import Agent.Hyperparameters as hp

import Main as m

def run():
    
    env, state_shape, action_shape = m.make_env(None)
    
    ddqAgent = m.load_agent(env, state_shape, action_shape)
    
    def in_loop(state, action, reward, next_state, done):
        ddqAgent.record_episode(state, action, reward, next_state, done)
        print(f"action {action}, reward: {reward}")            
    
    def end_episode():
        training_states, training_q_vectors = ddqAgent.process_episode()
        ddqAgent.train_offline(training_states, training_q_vectors)
        ddqAgent.qNet.model.save(hp.FILENAME)

    m.main(env, hp.TRAINING_NUM_EPISODES, ddqAgent.get_action_by_distribution, in_loop, end_episode)
    
    
run()
