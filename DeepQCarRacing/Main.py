import gymnasium as gym
import pygame
import tensorflow as tf

import DoubleDeepQAgent as ddqa
import Hyperparameters as hp
import ObservationProcessor as op

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
    state = op.ObservationProcessor.get_state(state)
    
    for i in range(num_iterations):
        
        if env.render_mode == "human":
            register_input()
        
        if quit:
            break
        
        action = human_action if interactive else get_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = op.ObservationProcessor.get_state(next_state)
        
        print('state: ', state)
        print('action: ', action, 'reward: ', reward)
    
        if train:
            ddqAgent.train(state, action, reward, next_state)
    
        state = next_state
        
        if terminated or truncated:
            state, info = env.reset()
            state = op.ObservationProcessor.get_state(state)

    env.close()
    
    if train:
        ddqAgent.qNet.model.save(hp.FILENAME)


def main(human_input = True, train = False, load_network = True):
    
    env = gym.make("CarRacing-v2", continuous = False, render_mode = "human") if human_input else\
          gym.make("CarRacing-v2", continuous = False)

    state = env.observation_space.sample()
    state = op.ObservationProcessor.get_state(state)
    state_shape = (len(state), )
    action_shape = (env.action_space.n, )
    
    ddqAgent = ddqa.DoubleDeepQAgent.load(env,
                                          state_shape,
                                          action_shape,
                                          hp.FILENAME,
                                          hp.EPOCHS,
                                          hp.TRAINING_ITERATION, 
                                          hp.WEIGHTS_TRANSFER_ITERATION, 
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
                                          hp.TRAINING_ITERATION, 
                                          hp.WEIGHTS_TRANSFER_ITERATION, 
                                          hp.GAMMA, 
                                          hp.EPSILON_DECAY)

    episodes = hp.EPISODES_TRAINING if train else\
               hp.EPISODES_RUNNING

    run(env, ddqAgent, episodes, train = train) 
