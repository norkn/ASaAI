import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import gymnasium as gym
from collections import deque


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
        block_vision = observation[46:67, 36:60]
        # Display the resized extracted region
        observation_resized = cv2.resize(block_vision, (10 * block_vision.shape[1], 10 * block_vision.shape[0]))
        cv2.imshow('Resized Extracted Region', observation_resized)

# Wait for a key press and close the windows
        cv2.waitKey(1)
        # Flatten the block_vision array and check if each pixel is grey
        block_vision_flat = [1 if np.all(pixel == pixel[0]) else 0 for row in block_vision for pixel in row]

        on_grass_left = observation[70:71, 46:47]
        on_grass_right = observation[70:71, 49:50]
        on_grass_left = 0 if ObservationProcessor._check_vision(on_grass_left) else 1
        on_grass_right = 0 if ObservationProcessor._check_vision(on_grass_right) else 1

        is_on_grass = 1 if on_grass_left == 1 and on_grass_right == 1 else 0

        return block_vision_flat + [is_on_grass]
    
    @staticmethod
    def get_state(observation):
        speed = ObservationProcessor.get_speed(observation)
        left_steering = ObservationProcessor.get_left_steering(observation)
        right_steering = ObservationProcessor.get_right_steering(observation)
        vision = ObservationProcessor.get_vision(observation)
        return [speed, left_steering, right_steering] + vision
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, input_size, output_size, epsilon_decay=0.99, min_epsilon=0.01, lr=0.1):
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
            action = np.random.choice(self.output_size-2)+1
            print("RANDOM ACTION: ", action)
            return action
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.q_network(state_tensor)
                action =  np.argmax(q_values.cpu().numpy())
                print(" MY CHOICE ACTION: ", action)
                return action

    def train(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([int(done)]).to(self.device)

        q_values = self.q_network(state_tensor)
        next_q_values = self.target_network(next_state_tensor)
        gammma = 0.95
        target_q_values = q_values.clone()
        target_q_values[action] = reward + gammma * torch.max(next_q_values).item() * (1 - done)

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        # Update target network
        if done:
            self.target_network.load_state_dict(self.q_network.state_dict())

        '''
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        '''

# Define your environment
# render_mode="human",
env = gym.make("CarRacing-v2", render_mode="human",continuous=False)
observation, info = env.reset()

# Create an instance of ObservationProcessor
processor = ObservationProcessor()

# Define hyperparameters
episodes = 5000
timesteps = 1000
fine_motor_skills = 1

# Define state size and action size based on your environment
#state_size = 1612 
state_size = 508  # Change this to the actual size
action_size = 5   # Change this to the actual size
epsilon = 0.4
#pathname = 'DQN_agent_007'
pathname = 'DQN_agent_3'
#pathname = 'DQN_agent_3_more_block_vision'
# Initialize DQNAgent
agent = DQNAgent(state_size, action_size)

# Load the saved state
checkpoint = torch.load(pathname + '.pth')

# Load model parameters and optimizer state
agent.q_network.load_state_dict(checkpoint['model_state_dict'])
agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
agent.set_epsilon(epsilon)


for episode in range(episodes):
    total_reward = 0
    observation, info = env.reset()
    if episode % 10 == 0:
        fine_motor_skills -= 1 if fine_motor_skills > 1 else 0
    
    # Decay epsilon
    if agent.epsilon > agent.min_epsilon:
        agent.epsilon *= agent.epsilon_decay
    for timestep in range(timesteps):
        state = processor.get_state(observation)
        
        if timestep % fine_motor_skills == 0: 
            action = agent.select_action(state)
            
        next_observation, reward, terminated, truncated, _ = env.step(action)

        #if processor.get_speed(observation) < 2:
        #    reward += -1
       
        #print(len(processor.get_vision(observation)[504]) )
        vision_list = processor.get_vision(observation)

        gray_vision_sum = sum(vision_list[:len(vision_list)-1])


        #print("Reward:" , reward)
        #print( "VISION" , (gray_vision_sum / 1000) )
        #print( "SPEED" , (processor.get_speed(observation)) )
        
        #print((gray_vision_sum / 10000) * processor.get_speed(observation))
        if (gray_vision_sum / 1000) * processor.get_speed(observation) == 0:
            reward += -2
        else:
            reward += (gray_vision_sum / 1000) * processor.get_speed(observation)

        
        #print("Reward after :" , reward)
        # Verwende List Comprehension, um die Zahlen zu summieren
        #anzahl_nullen = sum(1 for element in vision_list if element == 0)
        #negativ_vision_reward = anzahl_nullen / 1000
        #reward += -negativ_vision_reward


        next_state = processor.get_state(next_observation)
        agent.train(state, action, reward, next_state, terminated)

       
        total_reward += reward

        observation = next_observation
        #print(sum(vision_list[:504]))
        if terminated or truncated or total_reward < -300:
            break
    # Save the entire agent, including model parameters and optimizer state
    torch.save({
    'model_state_dict': agent.q_network.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
    'epsilon': agent.epsilon,
    'epsilon_decay': agent.epsilon_decay,
    'min_epsilon': agent.min_epsilon
    }, pathname + '.pth')        
    print("DQN SAVED!")
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon} LR: {agent.lr}")


env.close()