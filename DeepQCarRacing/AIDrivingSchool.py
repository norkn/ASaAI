import gymnasium as gym
import pygame

import tensorflow as tf

import DoubleDeepQAgent as ddqa
import ObservationProcessor as op

action = 0

def register_input():
    global quit, restart, action

    for event in pygame.event.get():

        if event.type == pygame.KEYDOWN:
            
            if event.key == pygame.K_LEFT:
                action = 2
            if event.key == pygame.K_RIGHT:
                action = 1
            if event.key == pygame.K_UP:
                action = 3
            if event.key == pygame.K_DOWN:
                action = 4
            
            if event.key == pygame.K_RETURN:
                restart = True
            if event.key == pygame.K_ESCAPE:
                quit = True

        if event.type == pygame.KEYUP:

            if event.key == pygame.K_LEFT:
                action = 0
            if event.key == pygame.K_RIGHT:
                action = 0
            if event.key == pygame.K_UP:
                action = 0
            if event.key == pygame.K_DOWN:
                action = 0

        if event.type == pygame.QUIT:
            quit = True

env = gym.make("CarRacing-v2", continuous = False, render_mode = "human")

FILENAME = 'CarRacingDDQWeights.keras'

ITERATIONS_TRAINING = 5000
ITERATIONS_RUNNING = 1000

LAYER_SIZES = [48, 24]
LAYER_ACTIVATIONS = ['relu', 'relu', 'linear']
LEARNING_RATE = 0.005

#one step is equivalent to about 0.029s in human experience, so 100 steps is about 3s
GAMMA = 0.98 #rewards 50 steps in the future get weighted at 36.4%, 100 steps in the future 13.2%
EPSILON_DECAY = 0.998 #after 1000 steps 13.5% chance of taking a random action

ddqAgent = ddqa.DoubleDeepQAgent(env, LAYER_SIZES, LAYER_ACTIVATIONS, tf.keras.initializers.Zeros(), LEARNING_RATE, GAMMA, EPSILON_DECAY)

#last_time = 0

quit = False

while not quit:

    state, info = env.reset()

    total_reward = 0.0
    steps = 0
    restart = False

    while True:
        
        #a = ddqAgent.get_action(state)
        register_input()
        
        print('action: ', action)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        ddqAgent.train(state, action, reward, next_state)

        state = next_state
        
        total_reward += reward

        #if steps % 1 == 0 or terminated or truncated:
            # print("\naction " + str([f"{x:+0.2f}" for x in a]))
            # print(f"step {steps} total_reward {total_reward:+0.2f}")
            # current_time = time.time()
            # print(current_time - last_time)
            # last_time = current_time
            #print('observation', op.ObservationProcessor.get_state(s))

        #steps += 1

        if terminated or truncated or restart or quit:
            break
        
env.close()

ddqAgent.qNet.model.save(FILENAME)
