import tensorflow as tf

LAYER_SIZES = [8*128, 4*128, 2*128, 2*64, 64]
LAYER_ACTIVATIONS = 5 * ['relu'] + ['linear']#5*[tf.keras.layers.LeakyReLU(alpha=0.01)]+['linear']

LEARNING_RATE = 0.001
LOSS = tf.keras.losses.MeanSquaredError()
OPTIMIZER = tf.keras.optimizers.Adam

NUM_BATCHES = 100
EPOCHS = 50

SAMPLE_SIZE = 3000
TRAINING_ITERATIONS = 1

#one step is equivalent to about 0.029s in human experience, ie 100 steps is about 3s
GAMMA = 0.98 #rewards 50 steps in the future get weighted at 36.4%, 100 steps in the future 13.2%
EPSILON_DECAY = 0.9998 #after 1000 steps (ie 30s) 81.9% chance of taking a random action

TRAINING_STEPS = 3000
RUNNING_STEPS = 1000

FILENAME = 'Savefiles/CarRacingDDQWeights.keras'
