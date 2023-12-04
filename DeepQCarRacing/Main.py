import gymnasium as gym
import tensorflow as tf

import DoubleDeepQAgent as ddqa

FILENAME = 'CarRacingDDQWeights.keras'

ITERATIONS_TRAINING = 5000
ITERATIONS_RUNNING = 1000

LAYER_SIZES = [24, 12]
LAYER_ACTIVATIONS = ['relu', 'relu', 'linear']
LEARNING_RATE = 0.001

EPSILON_DECAY = 0.9999

def run(env, ddqAgent, num_iterations, train = True):    
    state, info = env.reset()
    
    for i in range(num_iterations):
        
        if train:
            action = ddqAgent.get_action_epsilon_greedy(state)
            
            #first pick up speed to get things going
            if i < 100:
                action = 3
            
            next_state, reward, terminated, truncated, info = env.step(action)
        
            ddqAgent.train(state, action, reward, next_state)
        
            state = next_state
            
        else:
            action = ddqAgent.get_action(state)
            
            #first pick up speed to get things going
            if i < 100:
                action = 3
            
            print(action)
            state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            state, info = env.reset()

    env.close()
    
    if train:
        ddqAgent.qNet.model.save(FILENAME)


def mainTraining():
    env = gym.make("CarRacing-v2", continuous = False)
    ddqAgent = ddqa.DoubleDeepQAgent(env, LAYER_SIZES, LAYER_ACTIVATIONS, tf.keras.initializers.Zeros(), LEARNING_RATE, EPSILON_DECAY)  
    
    run(env, ddqAgent, ITERATIONS_TRAINING, train = True)
    
def mainContinueTraining():
    env = gym.make("CarRacing-v2", continuous = False)
    ddqAgent = ddqa.DoubleDeepQAgent.load(env, FILENAME, EPSILON_DECAY)
    
    run(env, ddqAgent, ITERATIONS_TRAINING, train = True)
    
    
def mainRun():
    env = gym.make("CarRacing-v2", continuous = False, render_mode = "human")
    ddqAgent = ddqa.DoubleDeepQAgent.load(env, FILENAME, EPSILON_DECAY)
    
    run(env, ddqAgent, ITERATIONS_RUNNING, train = False)
    
#mainTraining()
#mainContinueTraining()
mainRun()
