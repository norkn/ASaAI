# %% load imports und classes
#load imports und classes##########################################################################################################
#########################################################################################################################
#########################################################################################################################
import cv2
import numpy as np
import pickle
import gymnasium as gym
from collections import defaultdict
import keras
import tensorflow as tf
from tensorflow import keras
import random
import matplotlib.pyplot as plt
from npy_append_array import NpyAppendArray

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
            # colored pixel
            if i[0] != i[1] or i[1] != i[2]:
                return 0
            # black pixel (rand der welt)
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
            # check if the pixels are black
            elif sub_lst[0][0] == 0 and sub_lst[0][1] == 0 and sub_lst[0][2] == 0:
                # print("END OF THE WORLD AHEAD")
                binary_values.append(0)
            else:
                # Append 1 for grey, 0 for non-grey
                binary_values.append(1)
        return binary_values

    @staticmethod
    def get_speed(observation):
        gray_state = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[85:94, 12:13]
        # observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return Trainingsdata_generator._threshold_and_sum(cropped_state)

    @staticmethod
    def get_left_steering(observation):
        gray_state = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[89:90, 41:47]
        # observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return Trainingsdata_generator._threshold_and_sum(cropped_state)

    @staticmethod
    def get_right_steering(observation):
        gray_state = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[89:90, 49:55]
        # observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
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
        # print("STATE: ", state)
        return state

    @staticmethod
    def get_state(observation):
        speed = Trainingsdata_generator.get_speed(observation)
        left_steering = Trainingsdata_generator.get_left_steering(observation)
        right_steering = Trainingsdata_generator.get_right_steering(observation)
        vision = Trainingsdata_generator.get_vision(observation)
        return [speed, left_steering, right_steering] + list(vision)

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

        return [near_vision_left, near_vision_right, far_vision_left, far_vision_right, vision_stripe,
                is_on_grass] + vision_array

    @staticmethod
    def get_state(observation):

        speed = Agent_state_Processor.get_speed(observation)
        left_steering = Agent_state_Processor.get_left_steering(observation)
        right_steering = Agent_state_Processor.get_right_steering(observation)

        vision = Agent_state_Processor.get_vision(observation)

        return [speed, left_steering, right_steering] + list(vision)

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

def save_file(filedata, filename):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(filedata, file)
        print(f"Training data saved to {filename} successfully.")
    except Exception as e:
        print(f"Error occurred while saving training data to {filename}: {e}")

def process_and_save_training_data(training_data):
    q_table = defaultdict(lambda: np.zeros(5))  # actionshape = 5
    gamma = 0.98
    q_table_LR = 0.5
    q_value = 0
    for episode in training_data:
        for step in reversed(episode):
            state = step[0]
            action = step[1]
            reward = step[2]

            q_value = reward + gamma * q_value

            q_table[tuple(state)][action] = (1 - q_table_LR) * q_table[tuple(state)][action] + (q_table_LR) * q_value

    # Save training_data to a file using pickl
    q_table = dict(q_table)
    save_file(q_table, q_table_filename)


def get_dummy_action(state):
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

# Filenames
q_table_filename        = "Q_table_v2.npy"
trainings_data_filename = "training_dataset_1000_v2.npy"
model_filename          = "model_v2.keras"
rl_model_filename       = "RL_Agent_v2.keras"

dummy_reward_list_filename      = 'dummy_reward_list.npy'
pre_train_reward_list_filename  = "pre_train_reward_list.npy"
rl_reward_list_filename         = "rl_reward_list.npy"
#Reward Lists
training_data = []

dummy_reward_list =[]
pre_train_reward_list = []
rl_reward_list = []

aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'total': []}  # Dictionary welches die rewards pro episode abspeichert.
aggr_ep_model   = {'ep': [], 'avg': [],            'total': []}  # Dictionary welches die rewards pro episode abspeichert.
aggr_pretrain   = {'ep': [], 'avg': [],            'total': []}  # Dictionary welches die rewards pro episode abspeichert.


# RL - Variables
EPSILON = 0.20
EPSILON_DECAY = 0.95
GAMMA = 0.97
ACTION_SIZE = 5

trainings_info = []

# %% generate Trainingdata
##trainingsdaten generieren####################################################################################################
#########################################################################################################################
#########################################################################################################################
env = gym.make("CarRacing-v2", continuous=False)
observation, info = env.reset()

episodes = 50
timesteps = 1000 #1000

print("Starting the Dummy - loop")
# Loop too get trainingsdata
for episode in range(episodes):
    episode_info = []  # List to store information for each timestep in the episode
    total_reward = 0
    
    for timestep in range(timesteps):

        state = Trainingsdata_generator.get_state(observation)

        trainings_state = Agent_state_Processor.get_state(observation)  # len = 265

        action = get_dummy_action(state)
        observation, reward, terminated, truncated, info = env.step(action)

        next_state = Trainingsdata_generator.get_state(observation)
        trainings_next_state = Agent_state_Processor.get_state(observation)

        episode_info.append((trainings_state, action, reward, terminated or truncated))

        total_reward += reward
        dummy_reward_list.append(total_reward)

        if terminated or truncated:
            observation, info = env.reset()
            break
    training_data.append(episode_info)

    if not episodes % 1:
        average_reward_model = sum(dummy_reward_list[-timesteps:]) / len(dummy_reward_list[-timesteps:])
        aggr_pretrain['ep'].append(episode)
        aggr_pretrain['avg'].append(average_reward_model)
        aggr_pretrain['total'].append(total_reward)
    print(f"Episode {episode + 1} done!")

print("Finished the Dummy - loop")
np.save(dummy_reward_list_filename, dummy_reward_list)
np.save(trainings_data_filename, training_data)
#save_file(dummy_reward_list,'dummy_reward_list.pkl')
#save_file(training_data, trainings_data_filename)

#=========================================================================================================================#
#%% generate qtable with state action pair
#generate qtable with state action pair############################################################################################
#########################################################################################################################
#########################################################################################################################
training_data = np.load(trainings_data_filename,allow_pickle=True)
# q_werte der trainingsdaten berechnen
process_and_save_training_data(training_data)
#=========================================================================================================================#
# %% Pre-Training with dummy data
#Pre-Training with dummy data################################################################################################
#########################################################################################################################
#########################################################################################################################
state_shape = (265,)
action_shape = (5,)

layer_sizes = [64, 32, 16]
activation_functions = len(layer_sizes) * ['relu'] + ['linear']
init = tf.keras.initializers.RandomNormal(stddev=0.1, seed=0)

loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam
learning_rate = 0.01

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=state_shape))

for i in range(len(layer_sizes)):
    model.add(keras.layers.Dense(layer_sizes[i], activation=activation_functions[i], kernel_initializer=init))
model.add(keras.layers.Dense(action_shape[0], activation=activation_functions[-1], kernel_initializer=init))
model.compile(loss=loss, optimizer=optimizer(learning_rate=learning_rate))

q_table = np.load(q_table_filename, allow_pickle=True)

states = np.array(list(q_table))

q_values = np.array(list(q_table.values()))

num_batches = 100
epochs = 10

history = model.fit(  # returns history of the training
    states,
    q_values,
    batch_size=int(len(states) / num_batches),
    epochs=epochs,
    verbose=None
)

model.save(model_filename)

env = gym.make("CarRacing-v2", continuous=False)
observation, info = env.reset()
episodes  = 50       
timesteps = 1000

# Loop to train the agent further with trained model
for episode in range(episodes):
    total_reward = 0
    print(f"start Episode {episode + 1}")
    for timestep in range(timesteps):

        state = np.array(Agent_state_Processor.get_state(observation))

        action = np.argmax(model.predict(state.reshape(1, *state.shape), verbose=None)[0])

        observation, reward, terminated, truncated, info = env.step(action)

        next_state = np.array(Agent_state_Processor.get_state(observation))

        total_reward += reward
        pre_train_reward_list.append(total_reward)

        # Update the current state
        state = next_state

        if terminated or truncated:
            observation, info = env.reset()
            break

    if not episodes % 1:
        average_reward_model = sum(pre_train_reward_list[-timesteps:]) / len(pre_train_reward_list[-timesteps:])
        aggr_ep_model['ep'].append(episode)
        aggr_ep_model['avg'].append(average_reward_model)
        aggr_ep_model['total'].append(pre_train_reward_list)

#np.save('Model_Train_Rewards', pre_train_reward_list)
np.save("pre_train_reward_list.npy", pre_train_reward_list)
env.close()
#=========================================================================================================================#

#%%  RL-Training 
# RL-Training ############################################################################################
#########################################################################################################################
#########################################################################################################################
print("Start RL training")

env = gym.make("CarRacing-v2", continuous=False)
observation, info = env.reset()

model = keras.models.load_model(rl_model_filename)

episodes = 50# 500
timesteps = 1000 #100


# Loop to train the agent further with RL
for episode in range(episodes):
    print(f"start Episode {episode + 1}")

    episode_info = []  # List to store information for each timestep in the episode
    total_reward = 0

    if EPSILON > 0.01: EPSILON *= EPSILON_DECAY
    for timestep in range(timesteps):

        state = np.array(Agent_state_Processor.get_state(observation))
        if np.random.rand() < EPSILON:
            action = np.random.choice(ACTION_SIZE)
        else:
            action = np.argmax(model.predict(state.reshape(1, *state.shape), verbose=None)[0])
        observation, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(Agent_state_Processor.get_state(observation))

        episode_info.append((state, action, reward))

        total_reward += reward
        rl_reward_list.append(total_reward)

        state = next_state

        if terminated or truncated:
            observation, info = env.reset()
            break

    trainings_info.append(episode_info)

    if not episodes % 1:
        average_reward = sum(rl_reward_list[-timesteps:]) / len(rl_reward_list[-timesteps:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(rl_reward_list[-timesteps:]))
        aggr_ep_rewards['total'].append(total_reward)


    # RL train the agent
    if (episode + 1) % 5 == 0:
        c = 0
        print(f"Fit the Model on Episode {episode + 1} to {episode + 5}")
        for info in trainings_info:
            state_list = []
            target_list = []
            c += 1
            print(f"Info: {c}")
            for step in info:
                state_list.append(step[0])

            state_list = np.array(state_list)
            q_value = model.predict(state_list, verbose=None)


            for i in range(len(info) - 1):
                state, action, reward = info[i]
                target = q_value[i]
                target[action] = reward + GAMMA * np.max(q_value[i + 1])
                target_list.append(target)

        action = info[-1][1]
        reward = info[-1][2]

        target = np.zeros(5)
        target[action] = reward
        target_list.append(target)
        target_list = np.array(target_list)

        model.fit(state_list, target_list, batch_size=10, epochs=1, verbose=None)
        trainings_info = []
        print("Fitting Done")
        model.save(rl_model_filename)
        
env.close()
print("RL Training done")

np.save(rl_reward_list_filename,rl_reward_list)
#np.save('RL_Train_Rewards', ep_rewards)
#=========================================================================================================================#

# %% agent ueberpruefen und auswerten
#agent ueberpruefen und auswerten#################################################################################################
##########################################################################################################################
#########################################################################################################################
MinimumY = min(aggr_ep_rewards['min'])

dummy_reward_list = np.load(dummy_reward_list_filename, allow_pickle=True)
pre_trained_reward_list = np.load(pre_train_reward_list_filename,allow_pickle=True)
rl_reward_list = np.load(rl_reward_list_filename,allow_pickle=True)

x = np.arange(len(dummy_reward_list))
x1 = np.arange(len(pre_trained_reward_list))
x2 = np.arange(len(rl_reward_list))
plt.plot(x   , dummy_reward_list         , label="total_pretrain")
plt.plot(x1    , pre_trained_reward_list   , label="total_model")
plt.plot(x2  , rl_reward_list            , label="total_RL")
plt.ylim(MinimumY, 950)
plt.xlabel("episodes")
plt.ylabel("reward")
plt.legend(loc=4)
plt.show()

env = gym.make("CarRacing-v2", render_mode="human", continuous=False)

observation, info = env.reset()

episodes = 100
timesteps = 1000

# Loop to train the agent further with RL
for episode in range(episodes):
    total_reward = 0
    for timestep in range(timesteps):

        state = np.array(Agent_state_Processor.get_state(observation))

        action = np.argmax(model.predict(state.reshape(1, *state.shape),verbose=None)[0])

        observation, reward, terminated, truncated, info = env.step(action)

        next_state = np.array(Agent_state_Processor.get_state(observation))

        total_reward += reward

        # Update the current state
        state = next_state

        if terminated or truncated:
            observation, info = env.reset()
            break
env.close()
#================================================================================================#
# %%