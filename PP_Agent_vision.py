#%% load imports und classes 
import cv2
import numpy as np
import pickle
import gymnasium as gym
from collections import defaultdict
#import keras
import tensorflow as tf
from tensorflow import keras
import matplotlib as plt
#helper functions
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

def average_reward(reward_list):
    if not reward_list:
        return 0  # Return 0 for an empty list to avoid division by zero
    total_reward = sum(reward_list)
    average = total_reward / len(reward_list)
    return average
def save_file(filedata,filename):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(filedata, file)
        print(f"Training data saved to {filename} successfully.")
    except Exception as e:
        print(f"Error occurred while saving training data to {filename}: {e}")

def plot_reward_Comparison(lists, labels=None):
    if labels is None:
        labels = ['List 1', 'List 2', 'List 3']

    for i, data_list in enumerate(lists):
        plt.plot(data_list, label=labels[i])

    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Comparison')
    plt.show()

#Variables 
# Save training_data to a file using pickl
dummy_data_filename = "dummy_data_2_vision.pkl"
q_table_filename = "Q_table_2_vision.pkl"
model_filename = "model.keras"
RL_model_64_128_32 = "RL_model_64_128_32.keras"
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
        #on_grass_left = observation[63:64, 46:47]
        #on_grass_right = observation[63:64, 49:50]
        #on_grass_left_bottom = observation[77:78, 46:47]
        #on_grass_right_bottom = observation[77:78, 49:50]

        on_grass_left = Trainingsdata_generator._check_vision(on_grass_left)
        on_grass_right = Trainingsdata_generator._check_vision(on_grass_right)
        #on_grass_left_bottom = Trainingsdata_generator._check_gras(on_grass_left_bottom)
        #on_grass_right_bottom = Trainingsdata_generator._check_gras(on_grass_right_bottom)

        #is_on_grass = 1 if on_grass_left == 0 and on_grass_right == 0 and on_grass_left_bottom == 0 and on_grass_right_bottom == 0 else 0  # 1 == is on grass
        is_on_grass = 1 if on_grass_left == 0 and on_grass_right == 0 and on_grass_left == 0 and on_grass_right == 0 else 0  # 1 == is on grass

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
        vision_step = 2
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

#%% trainingsdaten generieren
env = gym.make("CarRacing-v2", continuous=False)

observation, info = env.reset()

episodes = 20
timesteps = 1000

training_data = []
reward_list_dummy = []
print("Starting the Dummy - loop")
# Loop too get trainingsdata
for episode in range(episodes):
    episode_info = []  # List to store information for each timestep in the episode
    total_reward = 0
    print(f"Episode {episode + 1} done!")
    for timestep in range(timesteps):
        
        state = Trainingsdata_generator.get_state(observation)

        trainings_state = Agent_state_Processor.get_state(observation) #len = 265
        
        action = get_dummy_action(state)
        observation, reward, terminated, truncated, info = env.step(action)

        next_state = Trainingsdata_generator.get_state(observation)
        trainings_next_state = Agent_state_Processor.get_state(observation)

        #episode_info.append((trainings_state, action, reward, terminated or truncated))
        episode_info.append((state, action, reward, terminated or truncated))
        total_reward += reward
        
        if terminated or truncated:
            observation, info = env.reset() 
            break
    reward_list_dummy.append(total_reward)
    training_data.append(episode_info)
env.close()
print("Finished the Dummy - loop")

save_file(training_data,dummy_data_filename)

average_reward_dummy = average_reward(reward_list_dummy)
save_file(average_reward_dummy, "average_reward_dummy.pkl")
print(f"Average Reward: {average_reward_dummy}")


#%% generate qtable with state action pairs

    
training_data = load_training_data(dummy_data_filename)


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
        #q_table_filename = "Q_table.pkl"
        try:
            with open(q_table_filename, 'wb') as file:
                pickle.dump(q_table, file)
            print(f"Training data saved to {q_table_filename} successfully.")
        except Exception as e:
            print(f"Error occurred while saving training data to {q_table_filename}: {e}")

# q_werte der trainingsdaten berechnen
process_and_save_training_data(training_data)

#%% agent an trainingsdaten trainieren lassen                            

state_shape =  (66,)
action_shape = (5,)

layer_sizes = [64,128,32]
activation_functions = len(layer_sizes) * ['relu'] + ['linear']
init = tf.keras.initializers.RandomNormal(stddev=0.1, seed=0)

loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam
learning_rate = 0.0001

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape = state_shape))

for i in range(len(layer_sizes)):
        model.add(keras.layers.Dense(layer_sizes[i], activation = activation_functions[i], kernel_initializer = init))
model.add(keras.layers.Dense(action_shape[0], activation = activation_functions[-1], kernel_initializer = init))
model.compile(loss = loss, optimizer = optimizer(learning_rate = learning_rate))

q_table = load_training_data(q_table_filename)

states = np.array(list(q_table))

q_values = np.array(list(q_table.values()))

num_batches = 100
epochs = 10

history = model.fit(            #returns history of the training
            states,
            q_values,
            batch_size = int(len(states) / num_batches),
            epochs = epochs,
            verbose = None
        )


model.save(RL_model_64_128_32)
'''
env = gym.make("CarRacing-v2",continuous=False)
observation, info = env.reset()
model = keras.models.load_model(RL_model_32_64_32)
episodes = 100
timesteps = 1000
reward_list_labeled = []

# Loop to train the agent further with RL
for episode in range(episodes):
    total_reward = 0
    print(f"Start Episode {episode+1}")
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
    reward_list_labeled.append(total_reward)
    print("Total Reward:", total_reward)
env.close()

average_reward_labeled = average_reward(reward_list_labeled)
print(f"Average Reward: {average_reward_labeled}")
'''
#%% agent via RL sich verbessern lassen

# Loop to train the agent further with RL
print("Start RL training")
env = gym.make("CarRacing-v2",render_mode = "human" ,continuous=False)
observation, info = env.reset()


model = keras.models.load_model(RL_model_64_128_32)

episodes = 60
timesteps = 1000

epsilon = 0.001
epsilon_decay = 0.98
gamma = 0.93
action_size = 5
trainings_info = []
reward_list_RL_agent = []

nRew = 0

reward = 0
last_reward = 0
# Loop to train the agent further with RL
for episode in range(episodes):
    episode_info = []  # List to store information for each timestep in the episode
    total_reward = 0

    nRew = 0
    print(f"Start Episode: {episode+1} Epsilon: {epsilon} ")

    if epsilon > 0.01 : epsilon *= epsilon_decay
    for timestep in range(timesteps):
        
        state = np.array(Agent_state_Processor.get_state(observation))
        

        if np.random.rand() < epsilon:
                action = np.random.choice(action_size-1)+1
                #print(f"Speed: {Trainingsdata_generator.get_speed(observation)}, Action: {action} Reward: {reward} OnGras {Trainingsdata_generator.get_state(observation)[3]}" )
        else:
            action = np.argmax(model.predict(state.reshape(1, *state.shape),verbose = None)[0])
            #print(f"Speed: {Trainingsdata_generator.get_speed(observation)}, Action: {action} Reward: {reward} OnGras {Trainingsdata_generator.get_state(observation)[3]}" )

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        '''
        if reward > 0:
            reward = Trainingsdata_generator.get_speed(observation) * 3

            nRew = 0

        if Trainingsdata_generator.get_state(observation)[3] == 1 and timestep > 20:
            nRew -= 1
            reward += nRew
        '''
       
        
        next_state = np.array(Agent_state_Processor.get_state(observation))

        episode_info.append((state, action, reward))

        # Update the current state
        state = next_state
        
        if terminated or truncated:
            observation, info = env.reset() 
            break
    reward_list_RL_agent.append(total_reward)
    trainings_info.append(episode_info)

    print(f"Total Reward: {total_reward}")
    
    # RL train the agent
    if (episode + 1) % 5 == 0:
        c = 0
        print(f"Fit the Model on Episode {episode + 1 - 5} to {episode}")
        for i in range(5):

            for info in trainings_info:
                state_list = []
                target_list = []

                for step in info:
                    state_list.append(step[0])

                state_list = np.array(state_list)
                q_value = model.predict(state_list, verbose = None)
                #print(f"QValue = {q_value} Shape = {q_value.shape}")
                
                for i in range(len(info)-1):
                    state,action,reward = info[i]
                    target = q_value[i]
                    target[action] = reward + gamma * np.max(q_value[i+1])
                    #print(f"Target = {target} Shape = {target.shape}")
                    target_list.append(target)

            action = info[-1][1]
            reward = info[-1][2]

            target = np.zeros(5)
            target[action] = reward

            target_list.append(target)
            target_list = np.array(target_list)
            #print(f"Target List = {target_list} Shape = {target_list.shape}")

            model.fit(state_list, target_list, batch_size=10, epochs=1, verbose=None)
        trainings_info = []
        keras.models.save_model(model,RL_model_64_128_32)
        print("Fitting Done")
env.close()

average_reward_RL_agent = average_reward(reward_list_RL_agent)
print(f"Average Reward: {average_reward_RL_agent}")
print("Training done")




#%% plot results
for data_list in enumerate(reward_list_RL_agent):
        plt.plot(data_list)

plt.legend()
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward Comparison')
plt.show()

#%% agent testen 
env = gym.make("CarRacing-v2", render_mode="human",continuous=False)
model = keras.models.load_model(RL_model_filename)
observation, info = env.reset()

episodes = 100
timesteps = 1000

# Loop to train the agent further with RL
for episode in range(episodes):
    
    total_reward = 0
    for timestep in range(timesteps):
        
        state = np.array(Agent_state_Processor.get_state(observation))
    
        action = np.argmax(model.predict(state.reshape(1, *state.shape),verbose = None)[0])

        observation, reward, terminated, truncated, info = env.step(action)

        next_state = np.array(Agent_state_Processor.get_state(observation))

        total_reward += reward

        # Update the current state
        state = next_state
        
        if terminated or truncated:
            observation, info = env.reset() 
            break
    print("Total Reward: ", total_reward)
env.close()
# %%
