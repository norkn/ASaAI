import gymnasium as gym
import numpy as np

import WrappedEnv as we

import DoubleDeepQAgent as ddqa
import Hyperparameters as hp

from npy_append_array import NpyAppendArray

def scripted_policy(state):
    action = 0
    if state[0] < 3: action = 3

    #far vision
    if state[5] == 0: action = 1 #turn right
    if state[6] == 0: action = 2 #turn left

    #near vision
    if state[3] == 0: action = 1 #turn right
    if state[4] == 0: action = 2 #turn left
    if state[0] == 0: action = 3

    return action

def main(env, num_iterations, get_action):   
    state, info = env.reset()

    total_reward = 0
    
    for i in range(num_iterations):
        
        action = get_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
    
        state = next_state
        
        if terminated or truncated:
            state, info = env.reset()

        total_reward +=  reward

    env.close()

    avg_reward_per_100 = 100 * (total_reward / num_iterations)

    print(f"steps: {num_iterations}, total reward: {total_reward}")
    print(f"average reward per 100 steps {avg_reward_per_100}")

    return avg_reward_per_100


def run():
    
    steps = hp.RUNNING_STEPS
    
    #########################
    np.random.seed(0)
    env2 = gym.make("CarRacing-v2", continuous = False,)# render_mode = "human")
    env2 = we.WrappedEnv(env2)

    scripted_result = main(env2, steps, scripted_policy)
    NpyAppendArray('Savefiles/scripted_results.npy',   delete_if_exists = False).append(np.array([scripted_result]))


    ##########################
    np.random.seed(0)
    env = gym.make("CarRacing-v2", continuous = False,)# render_mode = "human")
    env = we.WrappedEnv(env)
    
    state_shape = env.observation_space.shape    
    action_shape = (env.action_space.n, )
    
    ddqAgent = ddqa.DoubleDeepQAgent.load(env,
                                          state_shape,
                                          action_shape,
                                          hp.FILENAME,
                                          hp.NUM_BATCHES,
                                          hp.EPOCHS,
                                          hp.SAMPLE_SIZE, 
                                          hp.GAMMA, 
                                          hp.EPSILON_DECAY)


    rl_result = main(env, steps, ddqAgent.get_action)
    NpyAppendArray('Savefiles/rl_results.npy',   delete_if_exists = False).append(np.array([rl_result]))
    
run()
