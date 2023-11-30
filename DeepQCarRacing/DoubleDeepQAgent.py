import tensorflow as tf
import numpy as np

import DeepQNet as dqn

#experience replay

class DoubleDeepQAgent:
    
    def __init__(self, env, layer_sizes, activation_functions, init, learning_rate, epsilon_decay):
        self.env = env
        self.epsilon = 1.
        self.epsilon_decay = epsilon_decay
        
        self.numIterations = 0
        
        self.qNet = dqn.DeepQNet(env, layer_sizes, activation_functions, init, learning_rate)
        self.targetNet = dqn.DeepQNet(env, layer_sizes, activation_functions, init, learning_rate)
        
        self.training_states= []
        self.training_qValues = []
        
    def get_action(self, state):
        self.epsilon *= self.epsilon_decay
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.qNet.run(state))
            
            
        
    def train(self, state, action, reward, next_state):
        #record in batc
        target = reward + self.targetNet.run(next_state)   ### does reward get added to all components
        self.training_states.append(state)
        self.training_qValues.append(target)
        
        #fit weights every so often
        if self.numIterations % 100 == 0:
            self.qNet.train(self.training_states, self.training_qValues)
            self.training_states = []
            self.training_qValues = []
            
        if self.numIterations % 500 == 0:
            self.targetNet.set_weights(self.qNet.get_weights())
            
        self.numIterations += 1
            