import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

import DoubleDeepQAgent as ddqa

FILENAME = 'CarRacingDDQWeights.keras'

ITERATIONS = 100

LAYER_SIZES = [24, 12]
LAYER_ACTIVATIONS = ['relu', 'relu', 'linear']
LEARNING_RATE = 0.001

EPSILON_DECAY = 0.999

def run(env, ddqAgent, train = True):    
    state, info = env.reset()
    
    for _ in range(ITERATIONS):
        print(_)
        
        if train:
            action = ddqAgent.get_action_epsilon_greedy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
        
            ddqAgent.train(state, action, reward, next_state)
        
            state = next_state
            
        else:
            action = ddqAgent.get_action(state)
            print(action)
            state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            state, info = env.reset()

    env.close()
    
    ddqAgent.qNet.model.save(FILENAME)


def mainTraining():
    env = gym.make("CarRacing-v2", continuous = False)
    ddqAgent = ddqa.DoubleDeepQAgent(env, LAYER_SIZES, LAYER_ACTIVATIONS, tf.keras.initializers.HeUniform(), LEARNING_RATE, EPSILON_DECAY)  
    
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
    ddqAgent = ddqa.DoubleDeepQAgent(env, FILENAME, EPSILON_DECAY)
    
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
    ddqAgent = ddqa.DoubleDeepQAgent(env, FILENAME, EPSILON_DECAY)
    
    state, info = env.reset()
    
    for _ in range(200):
        action = ddqAgent.get_action(state)
        print(action)
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            state, info = env.reset()
            
    env.close()
    
mainTraining()
#mainContinueTraining()
#mainRun()