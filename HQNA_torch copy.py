import gymnasium as gym
import cv2
import numpy as np
from collections import defaultdict
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import gymnasium as gym
from collections import deque

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
    def _convert_to_binary(lst):
        binary_values = []

        for sub_lst in lst:
            #print("subList ", sub_lst)
            if sub_lst[0][0] != sub_lst[0][1] or sub_lst[0][1] != sub_lst[0][2]:
                binary_values.append(0)
            # Check if all numbers in the sublist are the same
            #is_grey = np.all(sub_lst == sub_lst[0])
            else:

            # Append 1 for grey, 0 for non-grey
                binary_values.append(1)
        return binary_values

    
    @staticmethod
    def get_speed(observation):
        gray_state = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[87:96, 12:13]
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
        cropped_state = gray_state[89:90, 49:55]
        #observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return ObservationProcessor._threshold_and_sum(cropped_state)

    @staticmethod
    def get_vision(observation):
        near_vision_left = observation[66:67, 44:45]
       
        near_vision_right = observation[66:67, 51:52]
        far_vision_left = observation[46:47, 44:45]
        far_vision_right = observation[46:47, 51:52]
        vision_stripe = observation[58:65, 48:49]
       
        on_grass_left = observation[70:71, 46:47]
        on_grass_right = observation[70:71, 49:50]

        near_vision_left = ObservationProcessor._check_vision(near_vision_left)
        near_vision_right = ObservationProcessor._check_vision(near_vision_right)
        far_vision_left = ObservationProcessor._check_vision(far_vision_left)
        far_vision_right = ObservationProcessor._check_vision(far_vision_right)
        vision_stripe = ObservationProcessor._convert_to_binary(vision_stripe)

        on_grass_left = ObservationProcessor._check_vision(on_grass_left)
        on_grass_right = ObservationProcessor._check_vision(on_grass_right)

        is_on_grass = 0 if on_grass_left == 0 and on_grass_right == 0 else 1 # 0 == is on grass

        state = [near_vision_left, near_vision_right, far_vision_left, far_vision_right]
       
        for i in vision_stripe:
            state.append(i)
        state.append(is_on_grass)

        #print("STATE: ", state)
      
        return state

    @staticmethod
    def get_state(observation):
        speed = ObservationProcessor.get_speed(observation)
        left_steering = ObservationProcessor.get_left_steering(observation)
        right_steering = ObservationProcessor.get_right_steering(observation)
        vision = ObservationProcessor.get_vision(observation)
        return [speed, left_steering, right_steering] + list(vision)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)
        self.fc2 = nn.Linear(4, 8)
        self.fc3 = nn.Linear(8, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, input_size, output_size, epsilon, epsilon_decay, min_epsilon, lr, gamma):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(input_size, output_size).to(self.device)
        self.target_network = DQN(input_size, output_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.lr = lr
        self.gammma = gamma
        self.optimizer = optim.Adam(self.q_network.parameters(), lr)
        self.loss_fn = nn.MSELoss()
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.output_size)
            #action = np.random.choice(self.output_size-2)+1
            #print("RANDOM ACTION: ", action)
            return action
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.q_network(state_tensor)
                action =  np.argmax(q_values.cpu().numpy())
                #print(" MY CHOICE ACTION: ", action)
                return action

    def train(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([int(done)]).to(self.device)

        q_values = self.q_network(state_tensor)
        next_q_values = self.target_network(next_state_tensor)
        target_q_values = q_values.clone()
        target_q_values[action] = reward + gamma * torch.max(next_q_values).item() * (1 - done)

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        # Update target network
        if done:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def load_model(self, file_path):
            # Load the state dict of the Q-network from the specified file
            self.q_network.load_state_dict(torch.load(file_path))
            # Set the target network to the same state as the Q-network
            self.target_network.load_state_dict(self.q_network.state_dict())
            # Set the networks to evaluation mode
            self.q_network.eval()
            self.target_network.eval()

def discretize_state(state):
    # You may need to adjust this function based on your state information
    # Convert continuous state to a tuple for the defaultdict
    return tuple(int(s) for s in state)


env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
observation, info = env.reset()

#filename = "DQN_AGENT_ADAM_64_128.pth"  #highscores Highscores: #1: 838, #2: 775, #3: 690
filename = "DQN_AGENT_ADAM_4_8.pth"  
action_size = 5
state_size = 15

epsilon = 0.01
epsilon_decay = 0.99
min_epsilon = 0.01
lr = 0.001
gamma = 0.995

dqn_agent = DQNAgent(state_size, action_size,epsilon, epsilon_decay, min_epsilon, lr, gamma)
dqn_agent.load_model(filename)
sessions = 10
episodes = 100
timesteps = 1000

# training relevant variables
highscore_this_episode = 0
highscores = []
top_trials_to_keep = 5  # Adjust as needed

# learning helper
fine_motor_skills = 1

processor = ObservationProcessor()

for session in range(sessions):
    print(f"\nSession {session + 1}/{sessions}")
    if session > 0:
        dqn_agent.set_epsilon(0.15)
        fine_motor_skills = max(1, fine_motor_skills - 1)

    for episode in range(episodes):
        
        #Highscores
        highscore_this_episode = 0
        highscores.sort(reverse=True)
        top_highscores = f"#1: {highscores[0]}, #2: {highscores[1]}, #3: {highscores[2]}" if len(highscores) > 3 else highscores

        total_reward = 0
        observation, info = env.reset()
     
        episode_info = []  # List to store information for each timestep in the episode

        for timestep in range(timesteps):
            state = processor.get_state(observation)

            if timestep % fine_motor_skills == 0:
                action = dqn_agent.select_action(state)

            # Take the action
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = processor.get_state(next_observation)

            # Extra Reward
            vision_list = processor.get_vision(observation)
            gray_vision_sum = sum(vision_list[:len(vision_list)-2])
            if ((gray_vision_sum / 50) * processor.get_speed(observation)) == 0:
                reward = -1
                if vision_list[-1] == 0:
                    reward = -20
                if action == 4:
                    reward = -50

            else:
                reward += (gray_vision_sum / 50) * processor.get_speed(observation)

            # Train the DQN agent
            #dqn_agent.train(state, action, reward, next_state, terminated)
            highscore_this_episode += (int)(reward)
            
            episode_info.append((state, action, reward, next_state, terminated))

            total_reward += reward
            observation = next_observation

            
            if terminated or truncated or timestep == timesteps - 1:
                highscores.append(highscore_this_episode)
                break

        print(f"""Episode {episode + 1}/{episodes}, 
                Reward this round: {highscore_this_episode},
                Highscores: {top_highscores}, 
                Epsilon: {dqn_agent.epsilon}, 
                FMS: {fine_motor_skills}""")
      # Train the DQN agent at the end of the episode
        for step_info in episode_info:
            state, action, reward, next_state, done = step_info
            dqn_agent.train(state, action, reward, next_state, done)
        # Decay epsilon
        if dqn_agent.epsilon > dqn_agent.min_epsilon:
            dqn_agent.epsilon *= dqn_agent.epsilon_decay
        # Save the agent's model after each session
        torch.save(dqn_agent.q_network.state_dict(), f"{filename}")

        # Update highscores
        highscores.append(total_reward)
        if len(highscores) > top_trials_to_keep:
            highscores.sort(reverse=True)
            highscores = highscores[:top_trials_to_keep]

    

    # Display overall progress after each session
    avg_score = sum(highscores) / len(highscores)
    print(f"\nOverall Progress for Session {session + 1}: Average Score: {avg_score}")
    avg_score = 0

env.close()