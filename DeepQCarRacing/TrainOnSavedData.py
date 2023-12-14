import tensorflow as tf
import gymnasium as gym

import WrappedEnv as we

import DoubleDeepQAgent as ddqa
import Hyperparameters as hp

env = gym.make("CarRacing-v2", continuous = False)

env = we.WrappedEnv(env)

state_shape = env.observation_space.shape

action_shape = (env.action_space.n, )

ddqAgent = ddqa.DoubleDeepQAgent(env, 
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

ddqa.DoubleDeepQAgent.load(env,
                           state_shape,
                           action_shape,
                           hp.FILENAME,
                           hp.EPOCHS,
                           hp.BATCH_SIZE, 
                           hp.TRAINING_ITERATIONS, 
                           hp.GAMMA, 
                           hp.EPSILON_DECAY)

ddqAgent.train_on_saved_data()

ddqAgent.qNet.model.save(hp.FILENAME)
print('weights saved, trained on saved data')
