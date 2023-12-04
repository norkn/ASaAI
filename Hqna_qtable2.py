import gymnasium as gym
import cv2
import numpy as np
from collections import defaultdict
import pickle

import seaborn as sns
import matplotlib.pyplot as plt



class ObservationProcessor:
    def __init__(self):
        pass

    @staticmethod
    def _threshold_and_sum(cropped_state):
        thresholded_state = np.where(cropped_state > 0, 1, 0)
        return np.sum(thresholded_state)

    @staticmethod
    def _check_vision(vision_array):
        for i in vision_array[0]:
            if i[0] != i[1] or i[1] != i[2]:
                return 0
        return 1

    @staticmethod
    def get_speed(observation):
        gray_state = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[88:93, 12:13]
        #observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return ObservationProcessor._threshold_and_sum(cropped_state)

    @staticmethod
    def get_left_steering(observation):
        gray_state = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[89:90, 41:47]
        #observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return ObservationProcessor._threshold_and_sum(cropped_state)

    @staticmethod
    def get_right_steering(observation):
        gray_state = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[89:90, 48:54]
        #observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return ObservationProcessor._threshold_and_sum(cropped_state)

    @staticmethod
    def get_vision(observation):
        near_vision_left = observation[66:67, 44:45]
       
        near_vision_right = observation[66:67, 51:52]
        far_vision_left = observation[46:47, 44:45]
        far_vision_right = observation[46:47, 51:52]
        vision_stripe = observation[50:51, 38:58]
        #vision_right = observation[70:71, 59:60]


        on_grass_left = observation[70:71, 46:47]
        on_grass_right = observation[70:71, 49:50]

        near_vision_left = ObservationProcessor._check_vision(near_vision_left)
        near_vision_right = ObservationProcessor._check_vision(near_vision_right)
        far_vision_left = ObservationProcessor._check_vision(far_vision_left)
        far_vision_right = ObservationProcessor._check_vision(far_vision_right)
        vision_stripe = ObservationProcessor._check_vision(vision_stripe)
        #vision_right = ObservationProcessor._check_vision(vision_right)

        on_grass_left = 0 if ObservationProcessor._check_vision(on_grass_left) else 1
        on_grass_right = 0 if ObservationProcessor._check_vision(on_grass_right) else 1

        is_on_grass = 1 if on_grass_left == 1 and on_grass_right == 1 else 0

        return [near_vision_left, near_vision_right, far_vision_left, far_vision_right, vision_stripe, is_on_grass]

    @staticmethod
    def get_state(observation):
        speed = ObservationProcessor.get_speed(observation)
        left_steering = ObservationProcessor.get_left_steering(observation)
        right_steering = ObservationProcessor.get_right_steering(observation)
        vision = ObservationProcessor.get_vision(observation)
        return [speed, left_steering, right_steering] + list(vision)

class QTable:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.q_table_length = 0
    def get_state_key(self, state):
        # Convert the state list to a tuple to use as a dictionary key
        return tuple(state)
    
    def get_q_table_len(self):
        return self.q_table_length
    
    def initialize_state(self, state):
        # Initialize the Q-values for a new state
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table_length += 1
            self.q_table[state_key] = np.zeros(self.action_size)

    def get_q_values(self, state):
        # Get Q-values for a state
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.initialize_state(state)
        return self.q_table[state_key]

    def update_q_value(self, state, action, new_q_value):
        # Update Q-value for a state-action pair
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.initialize_state(state)
        self.q_table[state_key][action] = new_q_value
        
    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            print("qTable saved: ", filename)
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            print("qTable loaded: ", filename)
            self.q_table = pickle.load(f)

    def plot_q_table(self, save_path=None):
        q_values = np.array(list(self.q_table.values()))
        sns.heatmap(q_values, cmap="YlGnBu", annot=True, fmt=".2f")
        plt.xlabel("Action")
        plt.ylabel("State")
        plt.title("Q-table")
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    


def discretize_state(state):
    # You may need to adjust this function based on your state information
    # Convert continuous state to a tuple for the defaultdict
    return tuple(int(s) for s in state)

    
#env = gym.make("LunarLander-v2", render_mode="human")
#env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
env = gym.make("CarRacing-v2",render_mode="human",continuous=False)
observation, info = env.reset()

#Q-table shape after episode 76: 1085 + 187 + 228 + 35 ´544 + 2
#qtable master 2 1183
#Episode: 76, Total Reward: -4806.44444444442, Epsilon: 0.09317615503395871 LR:  0.046588077516979354  
#filename = "Q_MASTER_2.pkl"
filename = "Q_MASTER_2_ADV.pkl"
action_size = 5
state_size = 1

actionlist = []
# Initialize Q-table as a defaultdict
q_table = QTable(state_size, action_size)
q_table.load_q_table(filename)
# Define hyperparameters
learning_rate = 0.1
discount_factor = 0.99
gamma = 0.9
epsilon = 0.01
episodes = 5000
timesteps = 1500


#learning helper 
fine_motor_skills = 10
gras_counter = 0

processor = ObservationProcessor()

for episode in range(episodes):
    total_reward = 0
    observation, info = env.reset()
    if epsilon > 0.01:
            epsilon *= discount_factor

    if learning_rate > 0.01:
            learning_rate *= discount_factor

    #imporve fine_motor_skills after 10 episodes
    if episode % 50 == 0:
        fine_motor_skills -= 1 if fine_motor_skills > 1 else 0
    
    episode_info = []
    gras_counter = 0
    for timestep in range(timesteps):

        state = processor.get_state(observation)
        
        discrete_state = discretize_state(state)

        # Update Q-value using the Q-learning update rule
        q_values = q_table.get_q_values(discrete_state)
        old_Q = np.copy(q_values)  # Store the old Q-values for comparison

        if timestep % fine_motor_skills == 0: 
            if np.random.rand() < epsilon:
                action = np.random.choice(action_size)
            else:
                action = np.argmax(q_table.get_q_values(discrete_state))

        # Take the action
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # Extra Reward
        vision_list = processor.get_vision(observation)
        gray_vision_sum = sum(vision_list[:len(vision_list)-1])
        if ((gray_vision_sum / 100) * processor.get_speed(observation)) == 0:
            reward += -1
        else:
            reward += (gray_vision_sum / 100) * processor.get_speed(observation)
        print("REWARD: " , reward)

        
        
        
        next_state =  processor.get_state(next_observation)
        discrete_next_state = discretize_state(next_state)

        # Update Q-value using the Q-learning update rule
        q_values[action] = q_values[action] + learning_rate * (
        reward + gamma * np.max(q_table.get_q_values(discrete_next_state)) - q_values[action]
        )

        new_Q = np.copy(q_values)  # Store the new Q-values for comparison

        q_table.update_q_value(discrete_state, action, q_values[action])

        
        total_reward += reward
        observation = next_observation
        
        
        if terminated or timestep == timesteps or total_reward < -200 :
            break  
    
        

    # Print Q-table dimensions after each episode
    print(f"Q-table shape after episode {episode + 1}: {q_table.q_table_length}")
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}", "LR: ", learning_rate)


    q_table.save_q_table(filename)
    

env.close()