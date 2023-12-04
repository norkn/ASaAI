import tensorflow as tf
import numpy as np

import DeepQNet as dqn

import ObservationProcessor as op

#experience replay

TRAINING_ITERATION = 10
WEIGHTS_TRANSFER_ITERATION = 50

class DoubleDeepQAgent:
    
    def __init__(self, env, layer_sizes, activation_functions, init, learning_rate, epsilon_decay):

        self.env = env
        
        state = self.env.observation_space.sample()
        print(op.ObservationProcessor.get_state(state))
        self.state_shape = tuple([len(op.ObservationProcessor.get_state(state))])
        
        self.epsilon = 1.
        self.epsilon_decay = epsilon_decay
        
        self.numIterations = 1
        
        self.qNet = dqn.DeepQNet(self.state_shape, layer_sizes, activation_functions, init, learning_rate)
        self.targetNet = dqn.DeepQNet(self.state_shape, layer_sizes, activation_functions, init, learning_rate)
        
        self.training_states= []
        self.training_qValues = []
    
    @staticmethod
    def load(env, path, epsilon_decay):
        model = DoubleDeepQAgent(env, 0, [], tf.keras.initializers.Zeros(), 0, epsilon_decay)
        
        model.qNet = dqn.DeepQNet.load(model.state_shape, epsilon_decay, path)
        model.targetNet = dqn.DeepQNet.load(model.state_shape, epsilon_decay, path)

        return model
        
    def get_action(self, state):
        state = op.ObservationProcessor.get_state(state)
        return np.argmax(self.qNet.run(state))
        
    def get_action_epsilon_greedy(self, state):
        self.epsilon *= self.epsilon_decay
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_action(state)
            
            
        
    def train(self, state, action, reward, next_state):
        state = op.ObservationProcessor.get_state(state)
        next_state = op.ObservationProcessor.get_state(next_state)
        
        target = reward + self.targetNet.run(next_state)
        
        self.training_states.append(state)
        self.training_qValues.append(target[0])
        print(target[0])
        
        if self.numIterations % TRAINING_ITERATION == 0:            
            training_states = np.array(self.training_states)
            training_qValues = np.array(self.training_qValues)
            
            self.qNet.train(training_states, training_qValues)
            self.training_states = []
            self.training_qValues = []
            
        if self.numIterations % WEIGHTS_TRANSFER_ITERATION == 0:
            self.targetNet.set_weights(self.qNet.get_weights())
            
        self.numIterations += 1
        
        
            