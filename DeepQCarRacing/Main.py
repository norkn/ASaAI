import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

import DoubleDeepQAgent as ddqa

FILENAME = 'CarRacingDDQWeights.keras'

def mainTraining():
    env = gym.make("CarRacing-v2") #use discrete actionspace
    ddqAgent = ddqa.DoubleDeepQAgent(env, [24, 12], ['relu', 'relu', 'linear'], tf.keras.initializers.HeUniform(), 0.001, 0.99)  
    
    state, info = env.reset()
    
    for _ in range(100):
        action = ddqAgent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        ddqAgent.train(state, action, reward, next_state)
        
        state = next_state
        
        if terminated or truncated:
            state, info = env.reset()

    env.close()
    
    ddqAgent.qNet.save(FILENAME)
    
    
def mainRun():
    env = gym.make("CarRacing-v2", render_mode = "human")
    ddqAgent = keras.models.load_model(FILENAME)
    
    state, info = env.reset()
    
    for _ in range(50):
        action = ddqAgent.get_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            state, info = env.reset()
            
    env.close()
    
mainTraining()