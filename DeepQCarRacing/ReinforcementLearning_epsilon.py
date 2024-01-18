import Agent.Hyperparameters as hp

import Main as m

def run():
    
    env, state_shape, action_shape = m.make_env(None)
    
    #ddqAgent = m.make_agent(env, state_shape, action_shape)
    ddqAgent = m.load_agent(env, state_shape, action_shape)
    
    total = 0         
    
    def in_loop(state, action, reward, next_state, done):
        nonlocal total
        total += reward
        ddqAgent.record_episode(state, action, reward, next_state, done)
    
    def end_episode():
        nonlocal total
        print("total reward at end of episode: ", total)
        total = 0

        ddqAgent.decay_epsilon()

        training_states, training_q_vectors = ddqAgent.process_training_data()
        ddqAgent.train_offline(training_states, training_q_vectors)
        ddqAgent.qNet.model.save(hp.FILENAME)

    m.main(env, hp.TRAINING_NUM_EPISODES, ddqAgent.get_action_epsilon_greedy, in_loop, end_episode)
    
    
run()
