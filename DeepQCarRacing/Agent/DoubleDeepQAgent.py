from collections import defaultdict

import numpy as np
from npy_append_array import NpyAppendArray

import Agent.DeepQNet as dqn

STATES_FILENAME = 'Savefiles/training_states.npy'
Q_VALUES_FILENAME = 'Savefiles/training_target_vectors.npy'
q_table_lerp_speed = 0.5
min_probability = 0.005
targetNet_update_step = 10

class DoubleDeepQAgent:
    
    def _reset_training_data(self):
        self.training_states      = []
        self.training_actions     = []
        self.training_rewards     = []
        self.training_next_states = []
        self.training_done        = []
    
    def __init__(self, 
        env,
        state_shape,
        action_shape,
        layer_sizes, 
        activation_functions, 
        init, 
        learning_rate,
        loss,
        optimizer,
        num_batches,
        epochs,
        sample_size,
        gamma, 
        epsilon_decay):

        self.env = env
        self.state_shape = state_shape
        self.action_shape = action_shape     
        
        if layer_sizes == None:
            self.qNet = None
            self.targetNet = None
        else:
            self.qNet      = dqn.DeepQNet(self.state_shape, self.action_shape, layer_sizes, activation_functions, init, learning_rate, loss, optimizer, num_batches, epochs)
            self.targetNet = dqn.DeepQNet(self.state_shape, self.action_shape, layer_sizes, activation_functions, init, learning_rate, loss, optimizer, num_batches, epochs)
        
        self._reset_training_data()

        self.sample_size = sample_size
    
        self.gamma = gamma
        self.epsilon = 1.
        self.epsilon_decay = epsilon_decay
        

    @staticmethod
    def load(
        env,
        state_shape,
        action_shape,
        path,
        num_batches,
        epochs,
        sample_size,
        gamma, 
        epsilon_decay):

        model = DoubleDeepQAgent(env, state_shape, action_shape, *[None]*6, num_batches, epochs, sample_size, gamma, epsilon_decay)
        
        model.qNet      = dqn.DeepQNet.load(path, num_batches, epochs)
        model.targetNet = dqn.DeepQNet.load(path, num_batches, epochs)

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
    
    def get_action_by_distribution(self, state):
        q_values = self.get_Q_values(state)

        min_element_index = np.argmin(q_values)

        shifted_q_values = q_values + abs(q_values[min_element_index])
        sum_shifted = np.sum(shifted_q_values)

        offset_for_min_element = min_probability / (1 - min_probability) * sum_shifted
        shifted_q_values[min_element_index] += offset_for_min_element
        sum_shifted += offset_for_min_element

        probability_distribution =  shifted_q_values / sum_shifted

        action = np.random.choice(range(self.action_shape[0]), 1, 
                                        p = probability_distribution)[0]

        return action
             
    def record_training_data(self, state, action, reward, next_state, done):
        self.training_states     .append(state)
        self.training_actions    .append(action)
        self.training_rewards    .append(reward)
        self.training_next_states.append(next_state)
        self.training_done       .append(done)
    
    def process_and_save_training_data(self):
        q_table = defaultdict(lambda: np.zeros(self.action_shape[0]))

        q_value = 0
        
        for i in reversed(range(len(self.training_states))):
            state  = self.training_states [i]
            action = self.training_actions[i]
            reward = self.training_rewards[i]
            done   = self.training_done   [i]

            if done:
                q_value = 0

            q_value = reward + self.gamma * q_value
            
            q_table[tuple(state)][action] = (1 - q_table_lerp_speed) * q_table[tuple(state)][action] + (q_table_lerp_speed) * q_value
            
        training_target_vectors = []
        for i in range(len(self.training_states)):
            training_target_vectors.append(np.array(q_table[ tuple(self.training_states[i]) ] ))

        training_states = np.array(self.training_states)
        training_target_vectors = np.array(training_target_vectors)
        
        NpyAppendArray(STATES_FILENAME,   delete_if_exists = False).append(training_states)
        NpyAppendArray(Q_VALUES_FILENAME, delete_if_exists = False).append(training_target_vectors)
        
        self._reset_training_data()

    def train_on_saved_data(self):
        training_states   = np.load(STATES_FILENAME,   mmap_mode="r")
        training_q_values = np.load(Q_VALUES_FILENAME, mmap_mode="r")

        self.qNet.train(training_states, training_q_values)
        
        self._reset_training_data()

    def train_on_new_data(self):
        training_targets = []
        
        for i in range(len(self.training_states)):
            state      = self.training_states     [i]
            action     = self.training_actions    [i]
            reward     = self.training_rewards    [i]
            next_state = self.training_next_states[i]

            if i % targetNet_update_step == 0:
                self.targetNet.set_weights(self.qNet.get_weights())
            
            target_vector = self.get_Q_values(state)
            target_vector[action] = reward + self.gamma * self.targetNet.run(next_state)[action]

            training_targets.append(target_vector)

        training_states = np.array(self.training_states)
        training_targets = np.array(training_targets)
        
        self.qNet.train(training_states, training_targets)
        
        self._reset_training_data()
        
            