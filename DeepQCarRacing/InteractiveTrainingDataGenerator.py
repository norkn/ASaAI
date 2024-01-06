import pygame

import Agent.DoubleDeepQAgent as ddqa
import Agent.Hyperparameters as hp

import Main as m

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
    
    env, state_shape, action_shape = m.make_env("human")
    
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

    m.main(env, hp.TRAINING_STEPS, register_input, ddqAgent.record_training_data, ddqAgent.process_and_save_training_data)
    
    
run()