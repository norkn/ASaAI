import gymnasium as gym

import pygame
import tensorflow as tf

import WrappedEnv as we

import DoubleDeepQAgent as ddqa
import Hyperparameters as hp

import Main

def register_input(state):

    for event in pygame.event.get():

        if event.type == pygame.KEYDOWN:
            
            if event.key == pygame.K_LEFT:
                register_input.action = 2
            if event.key == pygame.K_RIGHT:
                register_input.action = 1
            if event.key == pygame.K_UP:
                register_input.action = 3
            if event.key == pygame.K_DOWN:
                register_input.action = 4

        if event.type == pygame.KEYUP:

            if event.key == pygame.K_LEFT:
                register_input.action = 0
            if event.key == pygame.K_RIGHT:
                register_input.action = 0
            if event.key == pygame.K_UP:
                register_input.action = 0
            if event.key == pygame.K_DOWN:
                register_input.action = 0

    return register_input.action
register_input.action = 0


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

    Main.main(env, ddqAgent, steps, register_input)
    
    
run()
