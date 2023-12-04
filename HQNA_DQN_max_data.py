
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import gymnasium as gym
from collections import deque
import torch.nn.functional as F
import time

class ObservationProcessor:
    def __init__(self):
        pass

    @staticmethod
    def _threshold_and_sum(cropped_state):
        thresholded_state = np.where(cropped_state > 0, 1, 0)
        return np.sum(thresholded_state)

    @staticmethod
    def _check_vision(vision_array):
        return 1 if np.all(vision_array[0, :, 0] == vision_array[0, :, 1]) else 0

    @staticmethod
    def _get_cropped_gray(observation, row_range, col_range):
        gray_state = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[row_range[0]:row_range[1], col_range[0]:col_range[1]]
        return cropped_state

    @staticmethod
    def get_speed(observation):
        cropped_state = ObservationProcessor._get_cropped_gray(observation, (88, 95), (12, 13))
        return ObservationProcessor._threshold_and_sum(cropped_state)

    @staticmethod
    def _get_steering(observation, row_range, col_range):
        cropped_state = ObservationProcessor._get_cropped_gray(observation, row_range, col_range)
        return ObservationProcessor._threshold_and_sum(cropped_state)

    @staticmethod
    def get_left_steering(observation):
        return ObservationProcessor._get_steering(observation, (89, 90), (38, 47))

    @staticmethod
    def get_right_steering(observation):
        return ObservationProcessor._get_steering(observation, (89, 90), (48, 57))

    @staticmethod
    def _get_ABS(observation, row_range, col_range):
        cropped_state = ObservationProcessor._get_cropped_gray(observation, row_range, col_range)
        return ObservationProcessor._threshold_and_sum(cropped_state)

    @staticmethod
    def get_right_ABS(observation):
        return ObservationProcessor._get_ABS(observation, (89, 93), (22, 23))

    @staticmethod
    def get_left_ABS(observation):
        return ObservationProcessor._get_ABS(observation, (89, 93), (18, 19))

    @staticmethod
    def _get_gyro(observation, row_range, col_range):
        cropped_state = ObservationProcessor._get_cropped_gray(observation, row_range, col_range)
        return ObservationProcessor._threshold_and_sum(cropped_state)

    @staticmethod
    def get_right_gyro(observation):
        return ObservationProcessor._get_gyro(observation, (89, 90), (72, 76))

    @staticmethod
    def get_left_gyro(observation):
        return ObservationProcessor._get_gyro(observation, (89, 90), (68, 72))

    @staticmethod
    def get_vision(observation):
        block_vision = observation[:-12, 20:-20, :]
    
    # Aggregate features
        aggregated_features = []
        for channel in range(3):  # Assuming the last dimension is for color channels
        # Use np.sum to count the number of "on" pixels in each channel
            aggregated_features.append(np.sum(block_vision[:, :, channel]))

        # Convert aggregated features to binary (0 or 1) based on a threshold
        threshold = 100  # Adjust this threshold based on your observation
        aggregated_features = [1 if count > threshold else 0 for count in aggregated_features]

        on_grass_left = observation[70:71, 46:47]
        on_grass_right = observation[70:71, 49:50]
        on_grass_left = 0 if ObservationProcessor._check_vision(on_grass_left) else 1
        on_grass_right = 0 if ObservationProcessor._check_vision(on_grass_right) else 1
        is_on_grass = 1 if on_grass_left == 1 and on_grass_right == 1 else 0

        return aggregated_features + [is_on_grass]

    @staticmethod
    def get_state(observation):
        speed = ObservationProcessor.get_speed(observation)
        left_steering = ObservationProcessor.get_left_steering(observation)
        right_steering = ObservationProcessor.get_right_steering(observation)
        left_ABS = ObservationProcessor.get_left_ABS(observation)
        right_ABS = ObservationProcessor.get_right_ABS(observation)
        left_gyro = ObservationProcessor.get_left_gyro(observation)
        right_gyro = ObservationProcessor.get_right_gyro(observation)
        vision = ObservationProcessor.get_vision(observation)
        return [speed, left_steering, right_steering, left_ABS, right_ABS, left_gyro, right_gyro] + vision

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, input_size, output_size, epsilon_decay=0.99, min_epsilon=0.01, lr=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(input_size, output_size).to(self.device)
        self.target_network = DQN(input_size, output_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.lr = lr
        self.optimizer = optim.Adam(self.q_network.parameters(), lr)
        self.loss_fn = nn.MSELoss()
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = 0.95
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.output_size)
            print("RANDOM ACTION: ", action)
            return action
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.q_network(state_tensor)
                action =  np.argmax(q_values.cpu().numpy())
                print(" MY CHOICE ACTION: ", action)
                return action

    
    def train_batch(self, states, actions, rewards, next_states, dones):
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_network(states_tensor)
        next_q_values = self.target_network(next_states_tensor)
        gamma = 0.95
        target_q_values = q_values.clone()
        target_q_values[range(len(actions)), actions] = rewards_tensor + gamma * torch.max(next_q_values, dim=1).values * (1 - dones_tensor)

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        # Update target network
        if any(dones):
            self.target_network.load_state_dict(self.q_network.state_dict())

        '''
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        '''

# Define your environment
# render_mode="human",
env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
observation, info = env.reset()

# Create an instance of ObservationProcessor
processor = ObservationProcessor()

# Define hyperparameters
episodes = 5000
timesteps = 1000
fine_motor_skills = 10

# Define state size and action size based on your environment
state_size = 11  # Change this to the actual size
action_size = 5   # Change this to the actual size
epsilon = 0.9

pathname = 'DQN_agent_16_32'
# Initialize DQNAgent
agent = DQNAgent(state_size, action_size)

# Load the saved state
checkpoint = torch.load(pathname + '.pth')

# Load model parameters and optimizer state
agent.q_network.load_state_dict(checkpoint['model_state_dict'])
agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
agent.set_epsilon(epsilon)

for episode in range(episodes):
    start_time = time.time()  # Record the start time for the episode
    total_reward = 0
    observation, info = env.reset()
    if episode % 20 == 0:
        fine_motor_skills -= 1 if fine_motor_skills > 1 else 0
    
    # Decay epsilon
    if agent.epsilon > agent.min_epsilon:
        agent.epsilon *= agent.epsilon_decay

    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_next_states = []
    episode_dones = []

    for timestep in range(timesteps):
        state = processor.get_state(observation)

        if timestep % fine_motor_skills == 0:
            action = agent.select_action(state)

        next_observation, reward, terminated, truncated, _ = env.step(action)

        next_state = processor.get_state(next_observation)

        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_next_states.append(next_state)
        episode_dones.append(terminated)

        #vision_list = processor.get_vision(observation)
        #gray_vision_sum = sum(vision_list[:len(vision_list)-1])
        #print("GREY: " ,(gray_vision_sum / 10000), " SPEED: ",  processor.get_speed(observation))
        #if ((gray_vision_sum / 10000) * processor.get_speed(observation)) == 0:
        #    reward += -2
        #else:
        #    reward += (gray_vision_sum / 10000) * processor.get_speed(observation)
        #print("REWARD: " , reward)
        print("STATE: ", state)
        total_reward += reward

        observation = next_observation

        if terminated or truncated or total_reward < -100:
            break

    agent_traintime_start = time.time()
    agent.train_batch(episode_states, episode_actions, episode_rewards, episode_next_states, episode_dones)
    agent_traintime_end = time.time()
    print("TRAIN TIME: ", agent_traintime_end - agent_traintime_start)

    # Save the entire agent, including model parameters and optimizer state
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'epsilon_decay': agent.epsilon_decay,
        'min_epsilon': agent.min_epsilon
    }, pathname + '.pth')

    print("DQN SAVED!")
    end_time = time.time()  # Record the end time for the episode
    episode_time = end_time - start_time
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon} LR: {agent.lr}, Time: {episode_time:.2f} seconds")

env.close()