import gymnasium as gym

import WrappedEnv as we

import DoubleDeepQAgent as ddqa
import Hyperparameters as hp

def main(env, ddqAgent, num_iterations, get_action):   
    state, info = env.reset()
    
    for i in range(num_iterations):
        
        action = get_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)

        ddqAgent.record_training_data(state, action, reward, next_state, terminated or truncated)

        if i % hp.SAMPLE_SIZE == 0:
            ddqAgent.train_on_new_data()
    
        state = next_state
        
        if terminated or truncated:
            state, info = env.reset()

    env.close()
    
    ddqAgent.qNet.model.save(hp.FILENAME)


def run():
    
    env = gym.make("CarRacing-v2", continuous = False, render_mode = "human")
    env = we.WrappedEnv(env)
    
    state_shape = env.observation_space.shape    
    action_shape = (env.action_space.n, )
    
    ddqAgent = ddqa.DoubleDeepQAgent.load(env,
                                          state_shape,
                                          action_shape,
                                          hp.FILENAME,
                                          hp.NUM_BATCHES,
                                          hp.EPOCHS,
                                          hp.SAMPLE_SIZE, 
                                          hp.TRAINING_ITERATIONS, 
                                          hp.GAMMA, 
                                          hp.EPSILON_DECAY)

    steps = hp.TRAINING_STEPS

    main(env, ddqAgent, steps, ddqAgent.get_action_epsilon_greedy)
    
    
run()
