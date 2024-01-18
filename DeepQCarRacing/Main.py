import gymnasium as gym
import tensorflow as tf

import Agent.DoubleDeepQAgent as ddqa
import Agent.Hyperparameters as hp

from Environment import WrappedEnv as we

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

def nop(*args):
        return

def make_env(render_mode):
    env = gym.make(hp.ENV, render_mode = render_mode, continuous = False)
    env = we.WrappedEnv(env)
    
    state_shape = env.observation_space.shape    
    action_shape = (env.action_space.n, )
    
    return env, state_shape, action_shape

def make_agent(env, state_shape, action_shape):
    ddqAgent = ddqa.DoubleDeepQAgent(env, 
                                     state_shape,
                                     action_shape,
                                     hp.LAYER_SIZES, 
                                     hp.LAYER_ACTIVATIONS, 
                                     tf.keras.initializers.RandomNormal(stddev=0.1),
                                     hp.LEARNING_RATE,
                                     hp.LOSS,
                                     hp.OPTIMIZER,
                                     hp.NUM_BATCHES,
                                     hp.EPOCHS,
                                     hp.GAMMA, 
                                     hp.EPSILON_DECAY)

    return ddqAgent

def load_agent(env, state_shape, action_shape):
    ddqAgent = ddqa.DoubleDeepQAgent.load(env,
                                          state_shape,
                                          action_shape,
                                          hp.FILENAME,
                                          hp.NUM_BATCHES,
                                          hp.EPOCHS,
                                          hp.GAMMA, 
                                          hp.EPSILON_DECAY)

    return ddqAgent


def main(env, num_episodes, get_action, in_loop, end_episode):   
    total_reward = 0

    state, info = env.reset()

    terminated, truncated = False, False
    
    for i in range(num_episodes):
        for i in range(hp.MAX_STEPS_PER_EPISODE):
            action = get_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
        #####<
            in_loop(state, action, reward, next_state, terminated or truncated)
        ########>
            state = next_state

            total_reward +=  reward

            if terminated or truncated:
                state, info = env.reset()
                break

        end_episode()


    env.close()

    avg_reward_per_episode = total_reward / num_episodes
    return avg_reward_per_episode

