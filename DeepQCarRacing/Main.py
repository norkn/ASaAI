import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

import DoubleDeepQAgent as ddqa

FILENAME = 'CarRacingDDQWeights.keras'
ITERATIONS = 2000

def mainTraining():
    env = gym.make("CarRacing-v2", continuous = False)
    ddqAgent = ddqa.DoubleDeepQAgent(env, [24, 12], ['relu', 'relu', 'linear'], tf.keras.initializers.HeUniform(), 0.001, 0.99)  
    
    state, info = env.reset()
    
    for _ in range(ITERATIONS):
        print(_)
        action = ddqAgent.get_action_epsilon_greedy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        ddqAgent.train(state, action, reward, next_state)
        
        state = next_state
        
        if terminated or truncated:
            state, info = env.reset()

    env.close()
    
    ddqAgent.qNet.model.save(FILENAME)
    
def mainContinueTraining():
    env = gym.make("CarRacing-v2", continuous = False)
    ddqAgent = ddqa.DoubleDeepQAgent(env, FILENAME, 0.99)
    
    state, info = env.reset()
    
    for _ in range(ITERATIONS):
        print(_)
        action = ddqAgent.get_action_epsilon_greedy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        ddqAgent.train(state, action, reward, next_state)
        
        state = next_state
        
        if terminated or truncated:
            state, info = env.reset()

    env.close()
    
    ddqAgent.qNet.model.save(FILENAME)
    
    
def mainRun():
    env = gym.make("CarRacing-v2", continuous = False, render_mode = "human")
    ddqAgent = ddqa.DoubleDeepQAgent(env, FILENAME, 0.99)
    
    state, info = env.reset()
    
    for _ in range(200):
        action = ddqAgent.get_action(state)
        print(action)
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            state, info = env.reset()
            
    env.close()
    
#mainTraining()
#mainContinueTraining()
mainRun()