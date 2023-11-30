import tensorflow as tf
from tensorflow import keras
import numpy as np

import DeepQNet as dqn

#experience replay

TRAINING_ITERATION = 100
WEIGHTS_TRANSFER_ITERATION = 500

class DoubleDeepQAgent:
    
    def __init__(self, env, layer_sizes, activation_functions, init, learning_rate, epsilon_decay):
        self.env = env
        self.epsilon = 1.
        self.epsilon_decay = epsilon_decay
        
        self.numIterations = 1
        
        self.qNet = dqn.DeepQNet(env, layer_sizes, activation_functions, init, learning_rate)
        self.targetNet = dqn.DeepQNet(env, layer_sizes, activation_functions, init, learning_rate)
        
        self.training_states= []
        self.training_qValues = []
        
    def __init__(self, env, path, epsilon_decay):
        self.env = env
        self.epsilon = 1.
        self.epsilon_decay = epsilon_decay
        
        self.numIterations = 1
        
        self.qNet = dqn.DeepQNet(path)
        self.targetNet = dqn.DeepQNet(path)
        
        self.training_states= []
        self.training_qValues = []
        
    def get_action(self, state):
        return np.argmax(self.qNet.run(state))
        
    def get_action_epsilon_greedy(self, state):
        self.epsilon *= self.epsilon_decay
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.qNet.run(state))
            
            
        
    def train(self, state, action, reward, next_state):
        target = reward + self.targetNet.run(next_state)
        
        self.training_states.append(state.reshape(1, *state.shape))
        self.training_qValues.append(target)
        
        if self.numIterations % TRAINING_ITERATION == 0:
            print(len(self.training_states), len(self.training_qValues))
            print(self.training_states[0].shape, self.training_qValues[0].shape)
            
            self.training_states = np.array(self.training_states)
            self.training_qValues = np.array(self.training_qValues)
            
            self.qNet.train(self.training_states, self.training_qValues)
            self.training_states = []
            self.training_qValues = []
            
        if self.numIterations % WEIGHTS_TRANSFER_ITERATION == 0:
            self.targetNet.set_weights(self.qNet.get_weights())
            
        self.numIterations += 1
        
        
            