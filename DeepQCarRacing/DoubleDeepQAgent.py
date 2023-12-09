import numpy as np

import DeepQNet as dqn

class DoubleDeepQAgent:
    
    def _reset_training_samples(self):
        self.training_states = []
        self.training_actions = []
        self.training_rewards = []
        self.training_next_states = []
    
    def __init__(self, 
        env,
        state_shape,
        action_shape,
        layer_sizes, 
        activation_functions, 
        init, 
        learning_rate,
        epochs,
        training_iteration, 
        weights_transfer_iteration, 
        gamma, 
        epsilon_decay):

        self.env = env
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.epochs = epochs      
        
        if layer_sizes == None:
            self.qNet = None
            self.targetNet = None
        else:
            self.qNet = dqn.DeepQNet(self.state_shape, self.action_shape, layer_sizes, activation_functions, init, learning_rate, self.epochs)
            self.targetNet = dqn.DeepQNet(self.state_shape, self.action_shape, layer_sizes, activation_functions, init, learning_rate, self.epochs)
        
        self._reset_training_samples()

        self.training_iteration = training_iteration
        self.weights_transfer_iteration = weights_transfer_iteration
    
        self.gamma = gamma
        self.epsilon = 1.
        self.epsilon_decay = epsilon_decay
        
        self.numIterations = 1

    @staticmethod
    def load(
        env,
        state_shape,
        action_shape,
        path,
        epochs,
        training_iteration, 
        weights_transfer_iteration, 
        gamma, 
        epsilon_decay):

        model = DoubleDeepQAgent(env, state_shape, action_shape, *[None]*4, epochs, training_iteration, weights_transfer_iteration, gamma, epsilon_decay)
        
        model.qNet = dqn.DeepQNet.load(path, model.epochs)
        model.targetNet = dqn.DeepQNet.load(path, model.epochs)

        return model
    
    def get_Q_values(self, state):
        return self.qNet.run(state)
    
    def get_action(self, state):        
        return np.argmax(self.get_Q_values(state))
        
    def get_action_epsilon_greedy(self, state):
        self.epsilon *= self.epsilon_decay
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_action(state)
             
        
    def train(self, state, action, reward, next_state):             
        self.training_states.append(state)
        self.training_actions.append(action)
        self.training_rewards.append(reward)
        self.training_next_states.append(next_state)
        
        if self.numIterations % self.training_iteration == 0:

            training_targets = []

            for i in range(len(self.training_states)):
                state = self.training_states[i]
                action = self.training_actions[i]
                reward = self.training_rewards[i]
                next_state = self.training_next_states[i]

                target_vector = self.get_Q_values(state)[0]
                target_vector[action] = reward + self.gamma * self.targetNet.run(next_state)[0][action]

                training_targets.append(target_vector)

            training_states = np.array(self.training_states)
            training_targets = np.array(training_targets)
            
            self.qNet.train(training_states, training_targets)

            self._reset_training_samples()
            
        if self.numIterations % self.weights_transfer_iteration == 0:
            self.targetNet.set_weights(self.qNet.get_weights())
            
        self.numIterations += 1
        
        
            