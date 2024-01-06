import gymnasium as gym

import WrappedEnv as we

import DoubleDeepQAgent as ddqa
import Hyperparameters as hp

import Main

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


def run():
    
    env = gym.make("CarRacing-v2", continuous = False, render_mode = "human")
    env = we.WrappedEnv(env)
    
    state_shape = env.observation_space.shape    
    action_shape = (env.action_space.n, )
    
    ddqAgent = ddqa.DoubleDeepQAgent(env, 
                                     state_shape,
                                     action_shape,
                                     hp.LAYER_SIZES, 
                                     hp.LAYER_ACTIVATIONS, 
                                     None, 
                                     hp.LEARNING_RATE,
                                     hp.LOSS,
                                     hp.OPTIMIZER,
                                     hp.NUM_BATCHES,
                                     hp.EPOCHS,
                                     hp.SAMPLE_SIZE, 
                                     hp.GAMMA, 
                                     hp.EPSILON_DECAY)

    steps = hp.TRAINING_STEPS

    Main.main(env, ddqAgent, steps, scripted_policy)
    
    
run()
