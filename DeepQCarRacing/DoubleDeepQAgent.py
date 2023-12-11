import numpy as np

import DeepQNet as dqn

class DoubleDeepQAgent:
    
    def _reset_training_samples(self):
        self.training_states = []
        self.training_actions = []
        self.training_rewards = []
        self.training_next_states = []
        self.training_done = []
    
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
             
    def record_training_data(self, state, action, reward, next_state, done):
        self.training_states.append(state)
        self.training_actions.append(action)
        self.training_rewards.append(reward)
        self.training_next_states.append(next_state)
        self.training_done.append(done)
    
    def process_training_data(self):
        training_target_vectors = []

        q_value = 0
        
        for i in reversed(range(len(self.training_states))):
            state      = self.training_states     [i]
            action     = self.training_actions    [i]
            reward     = self.training_rewards    [i]
            done       = self.training_done       [i]

            if done:
                q_value = 0

            q_value = reward + self.gamma * q_value

            target_vector = self.get_Q_values(state)
            target_vector[action] = q_value

            training_target_vectors.append(target_vector)

        training_states = np.array(self.training_states)
        training_target_vectors = np.array(training_target_vectors)
        
        self.qNet.train(training_states, training_target_vectors)
        
        self._reset_training_samples()
        
            