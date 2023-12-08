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
class QTable:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.q_table_length = 0  # Initialize q_table_length to 0

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
            pickle.dump(self.q_table_length, f)  # Also save q_table_length

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            print("qTable loaded: ", filename)
            self.q_table = pickle.load(f)
            self.q_table_length = pickle.load(f)  # Load q_table_length

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


#,render_mode="human",
env = gym.make("CarRacing-v2",render_mode="human",continuous=False)
observation, info = env.reset()


#qtable master 2 1183
#Episode: 76, Total Reward: -4806.44444444442, Epsilon: 0.09317615503395871 LR:  0.046588077516979354  
#filename = "Q_MASTER_2.pkl"

filename = "Q_MASTER_2_ADV2.pkl" #Q-table shape after episode 457: 1641
action_size = 5
state_size = 1


# Initialize Q-table as a defaultdict
q_table = QTable(state_size, action_size)
q_table.load_q_table(filename)

# Define hyperparameters
learning_rate = 0.1
discount_factor = 0.98
gamma = 0.95
epsilon = 0.2
sessions = 6
episodes = 400
timesteps = 1000

#training releveant variables
highscore__this_episode = 0
highscores = 0
top_trials_to_keep = 4  # Adjust as needed
#learning helper 
fine_motor_skills = 6

# List to store information for each episode
all_episodes_info = []
# Initialize lists to store progress information
episode_rewards = []
session_averages = []



processor = ObservationProcessor()

for session in range(sessions):
    if session > 0:
        #session_averages = []  # Clear the list at the beginning of each session

        print("Change Hyperparameter:")
        discount_factor = 0.995
        epsilon = 0.20
        learning_rate /= 10
    

    for episode in range(episodes):

        total_reward = 0
        highscore__this_episode = 0
        observation, info = env.reset()

        if epsilon > 0.01:
            epsilon *= discount_factor

    
        #imporve fine_motor_skills after 10 episodes
        fine_motor_skills = (int) (epsilon * 10) if (int) (epsilon * 10) > 1 else 2
        
        episode_info = []
 
        for timestep in range(timesteps):

            state = processor.get_state(observation)
            #print("state:         ",  state)
            discrete_state = discretize_state(state)
            print("discrete_state:",  len(discrete_state))
        
        #in the first session the agent need to do the same action several times to improve learning
            if timestep % fine_motor_skills == 0: 
                if np.random.rand() < epsilon:
                    action = np.random.choice(action_size)
                else:
                    action = np.argmax(q_table.get_q_values(discrete_state))

        # Take the action
            next_observation, reward, terminated, truncated, info = env.step(action)

        #keep track of how good the episode was, trains only on the best 8/10 episodes
            highscore__this_episode += reward

        # Extra Reward
            vision_list = processor.get_vision(observation)
            gray_vision_sum = sum(vision_list[:len(vision_list)-2])
            if ((gray_vision_sum / 50) * processor.get_speed(observation)) == 0: # or vision_list[len(vision_list)-1] == 0
                reward = -1
                if(vision_list[len(vision_list)-1] == 0):
                    reward = -20
                if(action == 4):
                    reward = -50
                
            else:
                reward += (gray_vision_sum / 50) * processor.get_speed(observation)
            #print("REWARD: " , reward)

            next_state =  processor.get_state(next_observation)
            discrete_next_state = discretize_state(next_state)
            #print("state:         ",  next_state)
            
            #print("discrete_state:",  discrete_next_state)
        # Collect information for this time step
            step_info = {
            'state': discrete_state,
            'action': action,
            'reward': reward,
            'next_state': discrete_next_state,
            'terminated': terminated,
            'highscores': highscore__this_episode
            }
            episode_info.append(step_info)

            total_reward += reward
            observation = next_observation
        
        
            if terminated or timestep == timesteps or total_reward < -2000:
                break  

    
         # Print Q-table dimensions after each episode
        

        # Append episode reward to the list
        episode_rewards.append(highscore__this_episode)
        # Store the information for this episode
        all_episodes_info.append(episode_info)

        # After every 10 episodes, sort the episodes based on total_reward
        if (episode + 1) % 5 == 0:
            all_episodes_info.sort(key=lambda x: sum(step['reward'] for step in x), reverse=True)
    
            # Keep the top episodes for training
            top_episodes_info = all_episodes_info[:top_trials_to_keep]

            # Keep only the episodes with the best high scores
            top_episodes_info.sort(key=lambda x: max(step['highscores'] for step in x), reverse=True)
            best_highscore_episodes = top_episodes_info[:top_trials_to_keep]

            # Flatten the list of episodes for training
            training_data = [step for episode_info in best_highscore_episodes for step in episode_info]

            # Train your model using the training_data
            for step_info in training_data:
                discrete_state = step_info['state']
                print("State: ", discrete_state)
                action = step_info['action']
                reward = step_info['reward']
                discrete_next_state = step_info['next_state']
                highscores = step_info['highscores']
                  # Calculate and append the average reward for the session
                session_average = sum(episode_rewards[-episodes:]) / episodes
                session_averages.append(session_average)
                # Update Q-value using the Q-learning update rule
                q_values = q_table.get_q_values(discrete_state)
                old_q = q_values[action]  # Store the old Q-value for comparison
                q_values[action] = q_values[action] + learning_rate * (
                    reward + gamma * np.max(q_table.get_q_values(discrete_next_state)) - q_values[action]
                )
                new_q = q_values[action]  # Store the new Q-value for comparison
                q_table.update_q_value(discrete_state, action, q_values[action])

                print(f"Step Info: {step_info}")

            q_table.save_q_table(filename)
            training_data = []
            all_episodes_info = []  # Clear the episode info list after training
             # Calculate and append the average reward for the session
            session_average = sum(episode_rewards[-episodes:]) / episodes
            session_averages.append(session_average)

   
    
        # Print Q-table dimensions after each episode
        print(f"Q-table shape after episode {episode + 1}: {q_table.get_q_table_len()}")
        print(f"Episode: {episode + 1}, Total Reward: {highscores}, Epsilon: {epsilon}", "LR: ", learning_rate)
   
            # Plot progress after each session
    plt.plot(session_averages)
    plt.title('Average Reward per Session')
    plt.xlabel('Session')
    plt.ylabel('Average Reward')
    plt.show()
    

env.close()