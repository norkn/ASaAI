import gymnasium as gym

import pygame
import tensorflow as tf

import WrappedEnv as we

import DoubleDeepQAgent as ddqa
import Hyperparameters as hp

def get_action(state):
    action = 0
    if state[0] < 3: action = 3

    #far vision
    if state[5] == 0: action = 1 #turn right
    if state[6] == 0: action = 2 #turn left

    #near vision
    if state[3] == 0: action = 1 #turn right
    if state[4] == 0: action = 2 #turn left
    if state[0] == 0: action = 3

    return action

def run(env, ddqAgent, num_iterations, train = True):   
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


def main():
    
    env = gym.make("CarRacing-v2", continuous = False, render_mode = "human")

    env = we.WrappedEnv(env)
    
    state_shape = env.observation_space.shape
    
    action_shape = (env.action_space.n, )
    
    ddqAgent = ddqa.DoubleDeepQAgent(     env, 
                                          state_shape,
                                          action_shape,
                                          hp.LAYER_SIZES, 
                                          hp.LAYER_ACTIVATIONS, 
                                          tf.keras.initializers.Zeros(), 
                                          hp.LEARNING_RATE,
                                          hp.EPOCHS,
                                          hp.BATCH_SIZE, 
                                          hp.TRAINING_ITERATIONS, 
                                          hp.GAMMA, 
                                          hp.EPSILON_DECAY)

    steps = hp.TRAINING_STEPS

    run(env, ddqAgent, steps)
    
    
main()
