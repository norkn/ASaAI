import cv2
import numpy as np
import pickle
import gymnasium as gym
from collections import defaultdict

# trainingsdaten generieren
class Trainingsdata_generator:
    def __init__(self):
        pass

    @staticmethod
    def _threshold_and_sum(cropped_state):
        thresholded_state = np.where(cropped_state > 0, 1, 0)
        return np.sum(thresholded_state)

    @staticmethod
    def _check_vision(vision_array):
        for i in vision_array[0]:
            #colored pixel
            if i[0] != i[1] or i[1] != i[2]:
                return 0
            #black pixel (rand der welt)
            if i[0] == 0 and i[1] == 0 and i[2] == 0:
                return 0
        return 1

    @staticmethod
    def _convert_to_binary(lst):
        binary_values = []

        for sub_lst in lst:
            # Check if all numbers in the sublist are the same
            if sub_lst[0][0] != sub_lst[0][1] or sub_lst[0][1] != sub_lst[0][2]:
                binary_values.append(0)
            #check if the pixels are black
            elif sub_lst[0][0] == 0 and sub_lst[0][1] == 0 and sub_lst[0][2] == 0:
                #print("END OF THE WORLD AHEAD")
                binary_values.append(0)
            else:
            # Append 1 for grey, 0 for non-grey
                binary_values.append(1)
        return binary_values

    
    @staticmethod
    def get_speed(observation):
        gray_state = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[85:94, 12:13]
        #observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return Trainingsdata_generator._threshold_and_sum(cropped_state)

    @staticmethod
    def get_left_steering(observation):
        gray_state = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[89:90, 41:47]
        #observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return Trainingsdata_generator._threshold_and_sum(cropped_state)

    @staticmethod
    def get_right_steering(observation):
        gray_state = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[89:90, 49:55]
        #observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return Trainingsdata_generator._threshold_and_sum(cropped_state)

    @staticmethod
    def get_vision(observation):
        stripe_left = Trainingsdata_generator._convert_to_binary(observation[36:66, 44:45])
        stripe_right = Trainingsdata_generator._convert_to_binary(observation[36:66, 51:52])

        wing_left = Trainingsdata_generator._convert_to_binary(observation[70:71, 40:45])
        wing_right = Trainingsdata_generator._convert_to_binary(observation[70:71, 52:57])

        on_grass_left = observation[70:71, 46:47]
        on_grass_right = observation[70:71, 49:50]

        on_grass_left = Trainingsdata_generator._check_vision(on_grass_left)
        on_grass_right = Trainingsdata_generator._check_vision(on_grass_right)

        is_on_grass = 1 if on_grass_left == 0 and on_grass_right == 0 else 0  # 1 == is on grass

        state = [is_on_grass]
       
        for i in stripe_left:
            state.append(i)
        for i in stripe_right:
            state.append(i)
        for i in wing_left:
            state.append(i)
        for i in wing_right:
            state.append(i) 
        #  state[0] ,   state[1]     , state[2]        ,  state[3]     , state[4-34]                 , state[34-64]                 ,                                          
        # [speed    ,   left_steering, right_steering] + [is_on_grass] + list(stripe_left.flatten()) + list(stripe_right.flatten()) + list(wing_left.flatten()) + list(wing_right.flatten())
        #print("STATE: ", state)
        return state
    
    @staticmethod
    def get_state(observation):
        speed = Trainingsdata_generator.get_speed(observation)
        left_steering = Trainingsdata_generator.get_left_steering(observation)
        right_steering = Trainingsdata_generator.get_right_steering(observation)
        vision = Trainingsdata_generator.get_vision(observation)
        return [speed, left_steering, right_steering] + list(vision)

import numpy as np

class Agent_state_Processor:

    @staticmethod
    def _threshold_and_sum(cropped_state):
        thresholded_state = np.where(cropped_state > 0, 1, 0)
        
        return np.sum(thresholded_state)

    @staticmethod
    def _is_on_road(pixel):
        return int(pixel[0] == pixel[1] and pixel[1] == pixel[2] and pixel[0] > 0)

    @staticmethod
    def _get_value_from_pixels(observation, dX, dY):
        cropped_state = observation[dX[0]:dX[1], dY[0]:dY[1], :]
        gray_state = (cropped_state[:, :, 0] + cropped_state[:, :, 1] + cropped_state[:, :, 2]) / 3.

        return Agent_state_Processor._threshold_and_sum(gray_state)

    @staticmethod
    def get_speed(observation):
        return Agent_state_Processor._get_value_from_pixels(observation, (88, 93), (12, 13))

    @staticmethod
    def get_left_steering(observation):
        return Agent_state_Processor._get_value_from_pixels(observation, (89, 90), (41, 47))

    @staticmethod
    def get_right_steering(observation):
        return Agent_state_Processor._get_value_from_pixels(observation, (89, 90), (48, 54))

    @staticmethod
    def get_vision(observation):
        near_vision_left = observation[66][44]       
        near_vision_right = observation[66][51]

        far_vision_left = observation[46][44]
        far_vision_right = observation[46][51]

        vision_stripe = observation[50][48]

        on_grass_left = observation[70][46]
        on_grass_right = observation[70][49]

        near_vision_left = Agent_state_Processor._is_on_road(near_vision_left)
        near_vision_right = Agent_state_Processor._is_on_road(near_vision_right)

        far_vision_left = Agent_state_Processor._is_on_road(far_vision_left)
        far_vision_right = Agent_state_Processor._is_on_road(far_vision_right)

        vision_stripe = Agent_state_Processor._is_on_road(vision_stripe)

        on_grass_left = 1 - Agent_state_Processor._is_on_road(on_grass_left)
        on_grass_right = 1 - Agent_state_Processor._is_on_road(on_grass_right)

        is_on_grass = int(on_grass_left == 1 and on_grass_right == 1)
    
        vision_array = []
        vision_step = 6
        for x in range(int(len(observation[0]) / vision_step)):
            for y in range(int(len(observation) / vision_step)):
                pixel = Agent_state_Processor()._is_on_road(observation[y * vision_step][x * vision_step])
                vision_array.append(pixel)

        return [near_vision_left, near_vision_right, far_vision_left, far_vision_right, vision_stripe, is_on_grass] + vision_array

    @staticmethod
    def get_state(observation):

        speed = Agent_state_Processor.get_speed(observation)
        left_steering = Agent_state_Processor.get_left_steering(observation)
        right_steering = Agent_state_Processor.get_right_steering(observation)

        vision = Agent_state_Processor.get_vision(observation)

        return [speed, left_steering, right_steering] + list(vision)
   
def get_action(state):

    action = 0
    left = 0
    right = 0
    left_stripe = state[4:34]
    right_stripe = state[34:64]

    for feature in left_stripe:
        left += feature
    for feature in right_stripe:
        right += feature

    if (left < right) and left < 20: action = 1
    if (left > right) and right < 20: action = 2
    
    if state[0] < 3: action = 3
    
    return action

env = gym.make("CarRacing-v2", continuous=False)

observation, info = env.reset()

episodes = 2
timesteps = 50

training_data = []

# Loop too get trainingsdata
for episode in range(episodes):
    episode_info = []  # List to store information for each timestep in the episode
    gates = 0
    for timestep in range(timesteps):
        
        state = Trainingsdata_generator.get_state(observation)

        trainings_state = Agent_state_Processor.get_state(observation) #len = 265
        
        action = get_action(state)
        observation, reward, terminated, truncated, info = env.step(action)

        next_state = Trainingsdata_generator.get_state(observation)
        trainings_next_state = Agent_state_Processor.get_state(observation)

        episode_info.append((trainings_state, action, reward, terminated or truncated))

        # highscore
        if reward >= 3:
            gates += 1
        
        if terminated or truncated:
            observation, info = env.reset() 
            break
    #print(f"Gates round {episode}: {gates}")
    training_data.append(episode_info)

# Save training_data to a file using pickl
trainings_data_filename = "training_dataset_1000.pkl"
try:
    with open(trainings_data_filename, 'wb') as file:
        pickle.dump(training_data, file)
    print(f"Training data saved to {trainings_data_filename} successfully.")
except Exception as e:
    print(f"Error occurred while saving training data to {trainings_data_filename}: {e}")

env.close()


# Function to load training data from a file using pickle
def load_training_data(filename):
    """
    Load training data from a file using pickle.

    Parameters:
    - filename (str): The name of the file to load.

    Returns:
    - list: The loaded training data or an empty list if the file is not found.
    """
    try:
        print(f"Loading training data from {filename}...")
        with open(filename, 'rb') as file:
            loaded_data = pickle.load(file)
            print("Training data loaded successfully.")
            return loaded_data
    except FileNotFoundError:
        print(f"File {filename} not found. Initializing with an empty list.")
        return []
    
training_data = load_training_data(trainings_data_filename)


def process_and_save_training_data(training_data):
        q_table = defaultdict(lambda: np.zeros(5)) #actionshape = 5
        gamma = 0.98
        q_table_LR = 0.5
        q_value = 0
        for episode in training_data:
            for step in reversed(episode):
                state  = step[0]
                action = step[1]
                reward = step[2]
        
                q_value = reward + gamma * q_value
                
                q_table[tuple(state)][action] = (1 - q_table_LR) * q_table[tuple(state)][action] + (q_table_LR) * q_value
            
        # Save training_data to a file using pickl
        q_table = dict(q_table)
        q_table_filename = "Q_table.pkl"
        try:
            with open(q_table_filename, 'wb') as file:
                pickle.dump(q_table, file)
            print(f"Training data saved to {q_table_filename} successfully.")
        except Exception as e:
            print(f"Error occurred while saving training data to {q_table_filename}: {e}")

# q_werte der trainingsdaten berechnen
process_and_save_training_data(training_data)
# agent an trainingsdaten trainieren lassen



# agent via RL sich verbessern lassen

# agent ueberpruefen und auswerten