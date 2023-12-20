import gymnasium as gym

import WrappedEnv as we

import DoubleDeepQAgent as ddqa
import Hyperparameters as hp

import Main as m

def run():
    
    env = gym.make("CarRacing-v2", continuous = False, render_mode = "human")
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
                                          hp.TRAINING_ITERATIONS, 
                                          hp.GAMMA, 
                                          hp.EPSILON_DECAY)

    steps = hp.TRAINING_STEPS

    m.main(env, ddqAgent, steps, ddqAgent.get_action)
    
    
run()
