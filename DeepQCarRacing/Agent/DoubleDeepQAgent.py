import numpy as np
from npy_append_array import NpyAppendArray

from Utils import LambdaDict as ld

import Agent.DeepQNet as dqn

STATES_FILENAME = 'Savefiles/training_states.npy'
Q_VALUES_FILENAME = 'Savefiles/training_target_vectors.npy'
EPISODES_FILENAME = 'Savefiles/episodes.npy'
q_table_lerp_speed = 0.2
min_probability = 0.005

class DoubleDeepQAgent:
    
    def _reset_episode(self):
        self.episode = []
    
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
        
        self._reset_episode()
    
        self.gamma = gamma
        self.epsilon = .15
        self.epsilon_decay = epsilon_decay
        

    @staticmethod
    def load(
        env,
        state_shape,
        action_shape,
        path,
        num_batches,
        epochs,
        gamma, 
        epsilon_decay):

        model = DoubleDeepQAgent(env, state_shape, action_shape, *[None]*6, num_batches, epochs, gamma, epsilon_decay)
        
        model.qNet      = dqn.DeepQNet.load(path, num_batches, epochs)
        model.targetNet = dqn.DeepQNet.load(path, num_batches, epochs)

        return model
    
    def get_Q_values_batch(self, states):
        return self.qNet.run_batch(states)

    def get_Q_values(self, state):
        return self.qNet.run(state)
    
    def get_action(self, state):        
        return np.argmax(self.get_Q_values(state))

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        
    def get_action_epsilon_greedy(self, state):        
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
             
    def record_episode(self, state, action, reward, next_state, done):
        self.episode.append(np.array([*state, action, reward, *next_state, int(done)]))

    def save_episode(self):
        episode = np.array(self.episode)
        NpyAppendArray(EPISODES_FILENAME, delete_if_exists = False).append(episode)
        self._reset_episode()
    
    def _save_training_data(self, training_states, training_q_vectors):
        NpyAppendArray(STATES_FILENAME,   delete_if_exists = False).append(training_states)
        NpyAppendArray(Q_VALUES_FILENAME, delete_if_exists = False).append(training_q_vectors)

    def process_episode(self):
        state_len = self.state_shape[0]

        self.episode = self.episode[:state_len]

        states = np.array([self.episode[i][:state_len] for i in range(len(self.episode))])
        q_values = self.get_Q_values_batch(states)
        
        #q_table = ld.LambdaDict(lambda state: self.get_Q_values(np.array(state)))
        q_table = ld.LambdaDict(lambda state: 0)
        for i in range(len(states)):
            state = states[i]
            q_value = q_values[i]
            q_table[tuple(state)] = q_value

        q_value = 0

        training_states = []
        training_q_vectors = []
        
        for step in reversed(self.episode):
            state = step[:state_len]
            action = int(step[state_len])
            reward = float(step[state_len + 1])
            done = bool(step[-1])

            if done:#
                q_value = 0#

            q_value = reward + self.gamma * q_value

            q_table_row = q_table[tuple(state)]            
            q_table_row[action] = (1 - q_table_lerp_speed) * q_table_row[action] + (q_table_lerp_speed) * q_value
            
            training_states.append(state)
            training_q_vectors.append(np.array(q_table_row))

        training_states = np.array(training_states)
        training_q_vectors = np.array(training_q_vectors)

        return training_states, training_q_vectors

    def load_and_process_episode(self):
        self._reset_episode()
        
        self.episode = np.load(EPISODES_FILENAME, mmap_mode="r")

        return self.process_episode()

    def process_episode_and_save_training_data(self):
        training_states, training_q_vectors = self.process_episode()
        self._save_training_data(training_states, training_q_vectors)
        
    def train_offline(self, training_states, training_q_vectors):
        self.qNet.train(training_states, training_q_vectors)    

    def train_on_saved_data(self):
        training_states   = np.load(STATES_FILENAME,   mmap_mode="r")
        training_q_vectors = np.load(Q_VALUES_FILENAME, mmap_mode="r")

        self.train_offline(training_states, training_q_vectors)

    def train_on_new_data(self):
        training_states = []
        training_q_vectors = []
        
        for i in range(len(self.episode)):
            state, action, reward, next_state, done = self.episode[i]
            
            q_vector = self.get_Q_values(state)
            q_vector[action] = reward + self.gamma * self.targetNet.run(next_state)[action]

            training_states.append(state)
            training_q_vectors.append(q_vector)

            if (i + 1) % 100 == 0:
                self.targetNet.set_weights(self.qNet.get_weights())

                np_training_states = np.array(training_states)
                np_training_q_vectors = np.array(training_q_vectors)

                self.qNet.train(np_training_states, np_training_q_vectors)

                training_states = []
                training_q_vectors = []
        
        self._reset_episode()
        
            