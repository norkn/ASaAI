import tensorflow as tf
import numpy as np

import DeepQNet as dqn

import ObservationProcessor as op

#experience replay

TRAINING_ITERATION = 10
WEIGHTS_TRANSFER_ITERATION = 50

class DoubleDeepQAgent:
    
    def __init__(self, env, layer_sizes, activation_functions, init, learning_rate, gamma, epsilon_decay):

        self.env = env
        
        state = self.env.observation_space.sample()
        self.state_shape = tuple([len(op.ObservationProcessor.get_state(state))])
        
        self.action_shape = (5,)
        
        self.gamma = gamma
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
    
    def get_Q_values(self, state):
        processed_state = op.ObservationProcessor.get_state(state)
        
        return self.qNet.run(processed_state)
    
    def get_action(self, state):        
        return np.argmax(self.get_Q_values(state))
        
    def get_action_epsilon_greedy(self, state):
        self.epsilon *= self.epsilon_decay
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_action(state)
            
            
        
    def train(self, state, action, reward, next_state):
        processed_state = op.ObservationProcessor.get_state(state)
        processed_next_state = op.ObservationProcessor.get_state(next_state)
        
        rewards_vector = np.zeros(self.action_shape)
        rewards_vector[action] = reward
        target = rewards_vector + self.gamma * self.targetNet.run(processed_next_state)
        
        print('Q(s): ', self.get_Q_values(state))
        print("gamma * Q(s')", self.gamma * self.targetNet.run(processed_next_state))
        print('target: ', target)
        
        self.training_states.append(processed_state)
        self.training_qValues.append(target[0])
        
        if self.numIterations % TRAINING_ITERATION == 0:            
            training_states = np.array(self.training_states)
            training_qValues = np.array(self.training_qValues)
            
            self.qNet.train(training_states, training_qValues)
            self.training_states = []
            self.training_qValues = []
            
        if self.numIterations % WEIGHTS_TRANSFER_ITERATION == 0:
            self.targetNet.set_weights(self.qNet.get_weights())
            
        self.numIterations += 1
        
        
            