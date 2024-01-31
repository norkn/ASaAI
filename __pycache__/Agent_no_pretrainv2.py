#%% Imports
import cv2
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


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
            # green pixel
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
model_filename = "RL_Agent_vB0.keras"
#rl_reward_list_filename = ''
#rl_model_filename_template = "RL_Agent_v{}.keras"

# Reward Lists
rl_reward_list = []

# Network Architecture
state_shape = (265,)
action_shape = (5,)
layer_sizes = [512, 256, 128, 64, 32, 16]
activation_functions = ['relu', 'relu', 'relu', 'relu', 'relu', 'linear']


# RL - Variables
epsilon = 0.98
EPSILON_DECAY = 0.993
GAMMA = 0.98
ACTION_SIZE = 5

# Functions
def create_model(state_shape, action_shape, layer_sizes, activation_functions, initializer_stddev=0.1, seed=0, learning_rate=0.0001):
    init = keras.initializers.RandomNormal(stddev=initializer_stddev, seed=seed)
    loss = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=state_shape))

    for i in range(len(layer_sizes)):
        model.add(keras.layers.Dense(layer_sizes[i], activation=activation_functions[i], kernel_initializer=init))

    model.add(keras.layers.Dense(action_shape[0], activation=activation_functions[-1], kernel_initializer=init))
    model.compile(loss=loss, optimizer=optimizer(learning_rate=learning_rate))
    return model



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
#%%model creation
# Create and save the initial model
model = create_model(state_shape, action_shape, layer_sizes, activation_functions)
model.save(model_filename)

#%% rl training start
# RL-training
print("Start RL training")
env = gym.make("CarRacing-v2", render_mode = "human", continuous=False)   #render_mode = "human",
observation, info = env.reset()

episodes = 200
timesteps = 1000
model_iteration = 102

# Load the pre-trained RL model if available
#rl_model_filename = rl_model_filename_template.format(model_iteration)

model = keras.models.load_model(f"RL_Agent_vB{model_iteration}.keras")
#model_iteration = 102
# RL training loop

memory_buffer = {'states': [], 'actions': [], 'rewards': []}
processor = Agent_state_Processor()
trainer = Trainingsdata_generator()
total_reward = 0

for epoch in range(50):
    epsilon = 0.9
    for episode in range(episodes):
        
        total_reward = 0
        
        if epsilon > 0.10:
            epsilon *= EPSILON_DECAY

        for timestep in range(timesteps):
            state = np.array(processor.get_state(observation))

            if np.random.rand() < epsilon:
                    if trainer.get_speed(observation) == 5 and action == 3:
                        while action == 3:
                            action = np.random.choice(ACTION_SIZE)
                    elif trainer.get_speed(observation) == 4:
                        action = np.random.choice(ACTION_SIZE)
                    elif trainer.get_speed(observation) == 3:
                        action = np.random.choice(ACTION_SIZE - 2) + 1  # only gas and steering
                    elif trainer.get_speed(observation) < 3:
                        action = 3
            else:
                action = np.argmax(model.predict(state.reshape(1, *state.shape), verbose=None)[0])

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward


            if timestep > 15 and trainer.get_vision(observation)[0] == 1:
                reward = -2
                terminated = True

            next_state = np.array(processor.get_state(observation))
            
            # Store data in the memory buffer
            memory_buffer['states'].append(state)
            memory_buffer['actions'].append(action)
            memory_buffer['rewards'].append(reward)

            state = next_state

            if terminated or truncated:
                observation, info = env.reset()
                break
        print(f"Episode {episode + 1} done! , Epsilon: {epsilon}, Total Reward {total_reward}")        
        rl_reward_list.append(total_reward)
        
        # RL train the agent periodically using the memory buffer
        if (episode + 1) % 5 == 0:
            state_array = np.array(memory_buffer['states'])
            action_array = np.array(memory_buffer['actions'])
            reward_array = np.array(memory_buffer['rewards'])

            q_values = model.predict(state_array, verbose=None)
            target_values = q_values.copy()

            for i in range(len(state_array) - 1):
                target_values[i, action_array[i]] = reward_array[i] + GAMMA * np.max(q_values[i + 1])

        # Train the model with the accumulated data
            model.fit(state_array, target_values, batch_size=10, epochs=1, verbose=None)

            #model.save(f"true_rl_agent_iteration_{model_iteration}")

            # Clear the memory buffer
            memory_buffer = {'states': [], 'actions': [], 'rewards': []}
        if (episode + 1) % 100 == 0:
            model_iteration += 100
            model.save(f"RL_Agent_vB{model_iteration}.keras")  
            np.save(f"rl_reward_list_vB{model_iteration}.npy", rl_reward_list)
        

env.close()
print("RL Training done")

#%% rl triaining mid

# RL-training
print("Start RL training")
env = gym.make("CarRacing-v2", render_mode = "human", continuous=False)   #render_mode = "human",
observation, info = env.reset()

episodes = 1000
timesteps = 1000
model_iteration = 1000

# Load the pre-trained RL model if available
#rl_model_filename = rl_model_filename_template.format(model_iteration)

model = keras.models.load_model(f"RL_Agent_vB{model_iteration}.keras")
# RL training loop
memory_buffer = {'states': [], 'actions': [], 'rewards': []}
processor = Trainingsdata_generator()
total_reward = 0
for episode in range(episodes):
    

    total_reward = 0
    rand = np.random.rand()

    if epsilon > 0.01:
        epsilon *= EPSILON_DECAY

    for timestep in range(timesteps):
        state = np.array(processor.get_state(observation))

        if timestep % 4 == 0:
            rand = np.random.rand()

        if np.random.rand() < epsilon:
            if rand < 0.5:
                        action = 3
            elif rand > 0.90:
                if rand < 0.95:
                        action = 0
                else:
                        action = 4
            else:
                if rand > 0.7:
                        action = 1
                else:
                        action = 2

            '''
            if trainer.get_speed(observation) == 5 and action == 3:
                    while action == 3:
                        action = np.random.choice(ACTION_SIZE)
            elif trainer.get_speed(observation) == 4:
                    action = np.random.choice(ACTION_SIZE)
            elif trainer.get_speed(observation) == 3:
                    action = np.random.choice(ACTION_SIZE - 2) + 1  # only gas and steering
            elif trainer.get_speed(observation) < 3:
                    action = 3
            '''
            '''
            if trainer.get_speed(observation) == 5 and action == 3:
                while action == 3:
                    action = np.random.choice(ACTION_SIZE)
            elif trainer.get_speed(observation) == 4:
                action = np.random.choice(ACTION_SIZE)
            elif trainer.get_speed(observation) == 3:
                action = np.random.choice(ACTION_SIZE - 2) + 1  # only gas and steering
            elif trainer.get_speed(observation) < 3:
                action = 3
            
            if timestep > 30:
                if sum(trainer.get_state(observation)[28:33]) < sum(trainer.get_state(observation)[58:63]):
                    if np.random.rand() < 0.2:
                        action = np.random.choice(ACTION_SIZE)
                    else:
                        action = 1
        
                if sum(trainer.get_state(observation)[28:33]) > sum(trainer.get_state(observation)[58:63]):
                    if np.random.rand() < 0.2:
                        action = np.random.choice(ACTION_SIZE)
                    else:
                        action = 2

            '''
        else:
            action = np.argmax(model.predict(state.reshape(1, *state.shape), verbose=None)[0])

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if reward > 0:
            reward *= processor.get_speed(observation)

        if timestep > 30 and processor.get_vision(observation)[0] == 1:
            terminated = True

        next_state = np.array(processor.get_state(observation))
       
        # Store data in the memory buffer
        memory_buffer['states'].append(state)
        memory_buffer['actions'].append(action)
        memory_buffer['rewards'].append(reward)

        state = next_state

        if terminated or truncated:
            observation, info = env.reset()
            break
    print(f"Episode {episode + 1} done! , Epsilon: {epsilon}, Total Reward {total_reward}")        
    rl_reward_list.append(total_reward)
    
    # RL train the agent periodically using the memory buffer
    if (episode + 1) % 5 == 0:
        state_array = np.array(memory_buffer['states'])
        action_array = np.array(memory_buffer['actions'])
        reward_array = np.array(memory_buffer['rewards'])

        q_values = model.predict(state_array, verbose=None)
        target_values = q_values.copy()

        for i in range(len(state_array) - 1):
            target_values[i, action_array[i]] = reward_array[i] + GAMMA * np.max(q_values[i + 1])

       # Train the model with the accumulated data
        model.fit(state_array, target_values, batch_size=10, epochs=1, verbose=None)

        #model.save(f"true_rl_agent_iteration_{model_iteration}")

        # Clear the memory buffer
        memory_buffer = {'states': [], 'actions': [], 'rewards': []}
    if (episode + 1) % 50 == 0:
        model_iteration += 1
        model.save(f"RL_Agent_vB{model_iteration}.keras")  
        np.save(f"rl_reward_list_vB{model_iteration}.npy", rl_reward_list)
    

env.close()
print("RL Training done")
#%% Plot
plotlist = []
model_iteration = 100
while True:
    loadlist = np.load(f"rl_reward_list_vB{model_iteration}.npy")
    for i in loadlist:
        plotlist.append(i)
    model_iteration += 100

    if model_iteration == 2200:
        break

plt.plot(plotlist, label="total_RL")
plt.ylim(-200, 1000)
plt.xlabel("episodes")
plt.ylabel("reward")
plt.legend(loc=4)
plt.show()

#%% Some Display of Skill
env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
observation, info = env.reset()

model_iteration = 2200
model = keras.models.load_model(f"RL_Agent_v{model_iteration}.keras")
# model = model.load(rl_model_filename)

episodes = 1000
timesteps = 1000

print("Some Display of Skill")
# Loop to demonstrate the agent's skill
for episode in range(episodes):
    total_reward = 0
    for timestep in range(timesteps):
        state = np.array(processor.get_state(observation))
        action = np.argmax(model.predict(state.reshape(1, *state.shape), verbose=None)[0])
        observation, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(processor.get_state(observation))
        total_reward += reward

        # Update the current state
        state = next_state

        if terminated or truncated:
            observation, info = env.reset()
            break
env.close()

# %%
