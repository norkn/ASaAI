import tensorflow as tf

LAYER_SIZES = [2**10, 2**9, 2**8, 2**7, 2**6]
LAYER_ACTIVATIONS = 5 * ['relu'] + ['linear']

LEARNING_RATE = 0.001
LOSS = tf.keras.losses.MeanSquaredError()
OPTIMIZER = tf.keras.optimizers.Adam

NUM_BATCHES = 100
EPOCHS = 25

SAMPLE_SIZE = 1000

#one step is equivalent to about 0.029s in human experience, ie 100 steps is about 3s
GAMMA = 0.98 #rewards 50 steps in the future get weighted at 36.4%, 100 steps in the future 13.2%
EPSILON_DECAY = 0.998 #after 1000 steps (ie 30s) 81.9% chance of taking a random action

TRAINING_STEPS = 1000#00
RUNNING_STEPS = 1000

FILENAME = 'Savefiles/CarRacingDDQWeights.keras'