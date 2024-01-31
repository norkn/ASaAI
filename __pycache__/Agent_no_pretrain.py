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

        wing_left = Trainingsdata_generator._convert_to_binary(observation[70:71, 35:45])
        wing_right = Trainingsdata_generator._convert_to_binary(observation[70:71, 52:62])

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


# Filenames
model_filename          = "model_v2.keras"
rl_reward_list_filename = ''

#
training_data = []
trainings_info = []

#Reward Lists
rl_reward_list = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'total': []}  # Dictionary welches die rewards pro episode abspeichert.


#net architecture
state_shape = (66,)
action_shape = (5,)
layer_sizes = [16,32,16]
activation_functions = ['relu', 'relu', 'relu', 'linear']
model_iteration = 0

# RL - Variables
epsilon = 0.99
EPSILON_DECAY = 0.985
GAMMA = 0.97
ACTION_SIZE = 5


#functions
def create_model(state_shape, action_shape, layer_sizes, activation_functions, initializer_stddev=0.1, seed=0, learning_rate=0.01):
    init = tf.keras.initializers.RandomNormal(stddev=initializer_stddev, seed=seed)
    
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam
    
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=state_shape))

    for i in range(len(layer_sizes)):
        model.add(keras.layers.Dense(layer_sizes[i], activation=activation_functions[i], kernel_initializer=init))

    model.add(keras.layers.Dense(action_shape[0], activation=activation_functions[-1], kernel_initializer=init))
    model.compile(loss=loss, optimizer=optimizer(learning_rate=learning_rate))

    return model
#%% create new model

model = create_model(state_shape, action_shape, layer_sizes, activation_functions)
model.save(model_filename)

# %% RL-training
print("Start RL training")

env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
observation, info = env.reset()

episodes = 1000
timesteps = 1000 #100

rl_model_filename = f"RL_Agent_v399.keras"
model = keras.models.load_model(model_filename)

trainer = Trainingsdata_generator()
# Loop to train the agent further with RL
for episode in range(episodes):
    print(f"start Episode {episode + 1}, Epsilon: {epsilon}")

    episode_info = []  # List to store information for each timestep in the episode
    total_reward = 0

    if epsilon > 0.01: epsilon *= EPSILON_DECAY
    for timestep in range(timesteps):
        state = np.array(trainer.get_state(observation))
        if np.random.rand() < epsilon:

            if trainer.get_speed(observation) == 5:
                while action == 3:
                    action = np.random.choice(ACTION_SIZE)

            elif trainer.get_speed(observation) == 4:
                action = np.random.choice(ACTION_SIZE)

            elif trainer.get_speed(observation) == 3:
                action = np.random.choice(ACTION_SIZE-2)+1 #only gas and steering

            elif trainer.get_speed(observation) < 3:
                action = 3

        else:
            action = np.argmax(model.predict(state.reshape(1, *state.shape), verbose=None)[0])

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if timestep > 30 and trainer.get_vision(observation)[0] == 1:
            terminated = True
        next_state = np.array(Agent_state_Processor.get_state(observation))
        episode_info.append((state, action, reward))
        state = next_state
        if terminated or truncated:
            observation, info = env.reset()
            break
    rl_reward_list.append(total_reward)

    trainings_info.append(episode_info)


    # RL train the agent
    if (episode + 1) % 2 == 0:
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

    if (episode + 1) % 1 == 0:
        model_iteration += 1
        model.save(f"RL_Agent_v{model_iteration}.keras")  
env.close()
print("RL Training done")

np.save(f"rl_reward_list_v{episode}.npy",rl_reward_list)


plt.plot(rl_reward_list, label="total_RL")
plt.ylim(-200, 1000)
plt.xlabel("episodes")
plt.ylabel("reward")
plt.legend(loc=4)
plt.show()

env = gym.make("CarRacing-v2", render_mode="human", continuous=False)

observation, info = env.reset()

#model = model.load(rl_model_filename)

episodes = 1000
timesteps = 1000

print("Some Display of Skill")
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
# %%
