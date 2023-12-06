import numpy as np

import DeepQNet as dqn

import ObservationProcessor as op

#experience replay

TRAINING_ITERATION = 1000
WEIGHTS_TRANSFER_ITERATION = 1500

class DoubleDeepQAgent:
    
    def __init__(self, env, layer_sizes, activation_functions, init, learning_rate, gamma, epsilon_decay):

        self.env = env
        
        state = self.env.observation_space.sample()
        self.state_shape = (len(op.ObservationProcessor.get_state(state)), )
        
        self.action_shape = (env.action_space.n, )
        
        self.gamma = gamma
        self.epsilon = 1.
        self.epsilon_decay = epsilon_decay
        
        self.numIterations = 1
        
        if layer_sizes == None:
            self.qNet = None
            self.targetNet = None
        else:
            self.qNet = dqn.DeepQNet(self.state_shape, self.action_shape, layer_sizes, activation_functions, init, learning_rate)
            self.targetNet = dqn.DeepQNet(self.state_shape, self.action_shape, layer_sizes, activation_functions, init, learning_rate)
        
        self.training_states= []
        self.training_qValues = []
    
    @staticmethod
    def load(env, path, gamma, epsilon_decay):
        model = DoubleDeepQAgent(env, *[None]*4, gamma, epsilon_decay)
        
        model.qNet = dqn.DeepQNet.load(path)
        model.targetNet = dqn.DeepQNet.load(path)

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
        
        target_vector = self.get_Q_values(state)[0]
        target_vector[action] = reward + self.gamma * self.targetNet.run(processed_next_state)[0][action] ###########################
        
        self.training_states.append(processed_state)
        self.training_qValues.append(target_vector)
        
        if self.numIterations % TRAINING_ITERATION == 0:
            training_states = np.array(self.training_states)
            training_qValues = np.array(self.training_qValues)
            
            self.qNet.train(training_states, training_qValues)
            self.training_states = []
            self.training_qValues = []
            
        if self.numIterations % WEIGHTS_TRANSFER_ITERATION == 0:
            self.targetNet.set_weights(self.qNet.get_weights())
            
        self.numIterations += 1
        
        
            