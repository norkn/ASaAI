import gymnasium as gym

import pygame
import tensorflow as tf

import WrappedEnv as we

import DoubleDeepQAgent as ddqa
import Hyperparameters as hp

human_action = 0
interactive = False
quit = False

def register_input():
    global quit, human_action, interactive

    for event in pygame.event.get():

        if event.type == pygame.KEYDOWN:
            
            if event.key == pygame.K_SPACE:
                interactive = not interactive
            
            if event.key == pygame.K_LEFT:
                human_action = 2
            if event.key == pygame.K_RIGHT:
                human_action = 1
            if event.key == pygame.K_UP:
                human_action = 3
            if event.key == pygame.K_DOWN:
                human_action = 4
            
            if event.key == pygame.K_ESCAPE:
                quit = True

        if event.type == pygame.KEYUP:

            if event.key == pygame.K_LEFT:
                human_action = 0
            if event.key == pygame.K_RIGHT:
                human_action = 0
            if event.key == pygame.K_UP:
                human_action = 0
            if event.key == pygame.K_DOWN:
                human_action = 0

        if event.type == pygame.QUIT:
            quit = True

def run(env, ddqAgent, num_iterations, train = True):   
    global quit, interactive, human_action

    get_action = ddqAgent.get_action_epsilon_greedy if train else ddqAgent.get_action

    state, info = env.reset()
    
    for i in range(num_iterations):
        
        if env.render_mode == "human":
            register_input()
        
        if quit:
            break
        
        action = human_action if interactive else get_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print('state: ', state)
        print('action: ', action, 'reward: ', reward)
    
        ddqAgent.record_training_data(state, action, reward, next_state, terminated or truncated)
    
        state = next_state
        
        if terminated or truncated:
            state, info = env.reset()

    env.close()
    
    if train:
        print('training')
        ddqAgent.process_training_data()
        ddqAgent.qNet.model.save(hp.FILENAME)
        print('weights saved')


def main(human_input = True, train = False, load_network = True):
    
    env = gym.make("CarRacing-v2", continuous = False, render_mode = "human") if human_input else\
          gym.make("CarRacing-v2", continuous = False)

    env = we.WrappedEnv(env)
    
    state_shape = env.observation_space.shape
    
    action_shape = (env.action_space.n, )
    
    ddqAgent = ddqa.DoubleDeepQAgent.load(env,
                                          state_shape,
                                          action_shape,
                                          hp.FILENAME,
                                          hp.EPOCHS,
                                          hp.BATCH_SIZE, 
                                          hp.TRAINING_ITERATIONS, 
                                          hp.GAMMA, 
                                          hp.EPSILON_DECAY) if load_network else\
               ddqa.DoubleDeepQAgent(     env, 
                                          state_shape,
                                          action_shape,
                                          hp.LAYER_SIZES, 
                                          hp.LAYER_ACTIVATIONS, 
                                          tf.keras.initializers.Zeros(), 
                                          hp.LEARNING_RATE,
                                          hp.EPOCHS,
                                          hp.BATCH_SIZE, 
                                          hp.TRAINING_ITERATIONS, 
                                          hp.GAMMA, 
                                          hp.EPSILON_DECAY)

    steps = hp.TRAINING_STEPS if train else\
               hp.RUNNING_STEPS

    run(env, ddqAgent, steps, train = train) 
