import numpy as np
from npy_append_array import NpyAppendArray

from collections import defaultdict

import Agent.NeuralNetwork as dqn
import Agent.Hyperparameters as hp

class DeepQAgent:
    
    def reset_episode(self):
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
        
        self.reset_episode()
    
        self.gamma = gamma
        self.epsilon = .2
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

        model = DeepQAgent(env, state_shape, action_shape, *[None]*6, num_batches, epochs, gamma, epsilon_decay)
        
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
             
    def record_episode(self, state, action, reward, next_state, done):
        self.episode.append(np.array([*state, action, reward, *next_state, int(done)]))

    def save_episode(self):
        episode = np.array(self.episode)
        NpyAppendArray(hp.EPISODES_FILENAME, delete_if_exists = False).append(episode)
        self.reset_episode()

    def load_episode(self):
        episode = np.load(hp.EPISODES_FILENAME, mmap_mode="r")
        return episode

    def _get_sarnd(self, step):
        state_len = self.state_shape[0]

        state      = np.array(step[:state_len])
        action     = int     (step[state_len])
        reward     = float   (step[state_len + 1])
        next_state = np.array(step[-state_len - 1 : -1])
        done       = bool    (step[-1])

        return state, action, reward, next_state, done

    def _fill_q_dict(self, q_table, s):
            q_values = self.get_Q_values_batch(s)    
            for i in range(len(s)):
                state = s[i]
                q_value = q_values[i]
                q_table[tuple(state)] = q_value

            return q_table

    def _get_sq_from_q_dict(self, q_table):
        states = []
        q_vectors = []

        for key in q_table.keys():
            states.append(key)
            q_vectors.append(np.array(q_table[key]))

        states = np.array(states)
        q_vectors = np.array(q_vectors)

        return states, q_vectors

    def process_steps(self, steps):
        states      = np.array([self._get_sarnd(step)[0] for step in steps])
        next_states = np.array([self._get_sarnd(step)[3] for step in steps])

        q_table = defaultdict(lambda: np.zeros(self.action_shape))
        q_table = self._fill_q_dict(q_table, states)
        q_table = self._fill_q_dict(q_table, next_states)

        for step in steps:
            state, action, reward, next_state, done = self._get_sarnd(step)

            q_value = reward + self.gamma * np.max(q_table[tuple(next_state)])
            if done:
                q_value = reward

            q_table_row = q_table[tuple(state)]            
            q_table_row[action] = (1 - hp.Q_TABLE_LERP_SPEED) * q_table_row[action] + (hp.Q_TABLE_LERP_SPEED) * q_value
            
        states, q_vectors = self._get_sq_from_q_dict(q_table)

        return states, q_vectors

    def fit(self, states, q_vectors):
        self.qNet.train(states, q_vectors)    

    def fit_to_measured_q_values(self, sample_size):
        episode = np.load(hp.EPISODES_FILENAME, mmap_mode="r")
        episode = episode[:sample_size]

        states = np.array([self._get_sarnd(step)[0] for step in steps])

        q_table = defaultdict(lambda: np.zeros(self.action_shape))
        q_table = self._fill_q_dict(q_table, states)
        
        q_value = 0
        q_vectors = []

        for step in reversed(self.episode):
            state, action, reward, _, done = self._get_sarnd(step)

            if done:
                q_value = 0

            q_value = reward + self.gamma * q_value

            q_table_row = q_table[tuple(state)]
            q_table_row[action] = (1 - hp.Q_TABLE_LERP_SPEED) * q_table_row[action] + hp.Q_TABLE_LERP_SPEED * q_value
            
            q_vectors.append(np.array(q_table_row))

        states = [s for s in reversed(states)]
        states = np.array(states)
        q_vectors = np.array(q_vectors)

        self.fit(states, q_vectors)       
            