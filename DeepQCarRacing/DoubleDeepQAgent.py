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
        batch_size, 
        training_iterations, 
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

        self.batch_size = batch_size
        self.training_iterations = training_iterations
    
        self.gamma = gamma
        self.epsilon = 1.
        self.epsilon_decay = epsilon_decay
        

    @staticmethod
    def load(
        env,
        state_shape,
        action_shape,
        path,
        epochs,
        batch_size, 
        training_iterations, 
        gamma, 
        epsilon_decay):

        model = DoubleDeepQAgent(env, state_shape, action_shape, *[None]*4, epochs, batch_size, training_iterations, gamma, epsilon_decay)
        
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
             
    def record_training_data(self, state, action, reward, next_state):
        self.training_states.append(state)
        self.training_actions.append(action)
        self.training_rewards.append(reward)
        self.training_next_states.append(next_state)    
    
    def train(self):
        for i in range(self.training_iterations): #training_loops = episodes_training / (training_iteration == batchsize)
            training_states_batch      = self.training_states     [i * self.batch_size : (i + 1) * self.batch_size]
            training_actions_batch     = self.training_actions    [i * self.batch_size : (i + 1) * self.batch_size]
            training_rewards_batch     = self.training_rewards    [i * self.batch_size : (i + 1) * self.batch_size]
            training_next_states_batch = self.training_next_states[i * self.batch_size : (i + 1) * self.batch_size]

            training_targets = []
            
            for k in range(len(training_states_batch)):
                state      = training_states_batch     [k]
                action     = training_actions_batch    [k]
                reward     = training_rewards_batch    [k]
                next_state = training_next_states_batch[k]

                target_vector = self.get_Q_values(state)[0]
                target_vector[action] = reward + self.gamma * self.targetNet.run(next_state)[0][action]

                training_targets.append(target_vector)

            training_states_batch = np.array(training_states_batch)
            training_targets = np.array(training_targets)
            
            self.qNet.train(training_states_batch, training_targets)
            self.targetNet.set_weights(self.qNet.get_weights())
        
        self._reset_training_samples()
        
            