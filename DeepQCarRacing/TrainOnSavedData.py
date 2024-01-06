import tensorflow as tf

import Agent.DoubleDeepQAgent as ddqa
import Agent.Hyperparameters as hp

import Main as m

env, state_shape, action_shape = m.make_env(None)

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
                                 hp.SAMPLE_SIZE, 
                                 hp.GAMMA, 
                                 hp.EPSILON_DECAY)

# ddqAgent = ddqa.DoubleDeepQAgent.load(env,
                        #    state_shape,
                        #    action_shape,
                        #    hp.FILENAME,
                        #    hp.NUM_BATCHES
                        #    hp.EPOCHS,
                        #    hp.SAMPLE_SIZE, 
                        #    hp.TRAINING_ITERATIONS, 
                        #    hp.GAMMA, 
                        #    hp.EPSILON_DECAY)

ddqAgent.train_on_saved_data()

ddqAgent.qNet.model.save(hp.FILENAME)
print('weights saved, trained on saved data')
