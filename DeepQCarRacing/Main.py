def main(env, ddqAgent, num_iterations, get_action):   
    state, info = env.reset()
    
    for i in range(num_iterations):
        
        action = get_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
    
        ddqAgent.record_training_data(state, action, reward, next_state, terminated or truncated)
    
        state = next_state
        
        if terminated or truncated:
            state, info = env.reset()

    env.close()
    
    ddqAgent.process_and_save_training_data()
