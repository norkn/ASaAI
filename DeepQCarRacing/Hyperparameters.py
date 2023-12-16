import tensorflow as tf

LAYER_SIZES = [8*128, 4*128, 2*128, 2*64, 64]
LAYER_ACTIVATIONS = 5 * ['relu'] + ['linear']#5*[tf.keras.layers.LeakyReLU(alpha=0.01)]+['linear']#3*['linear']
LEARNING_RATE = 0.00001

EPOCHS = 100

BATCH_SIZE = 30000
TRAINING_ITERATIONS = 1

#one step is equivalent to about 0.029s in human experience, ie 100 steps is about 3s
GAMMA = 0.98 #rewards 50 steps in the future get weighted at 36.4%, 100 steps in the future 13.2%
EPSILON_DECAY = 0.9998 #after 1000 steps 13.5% chance of taking a random action

TRAINING_STEPS = 30000#BATCH_SIZE * TRAINING_ITERATIONS
RUNNING_STEPS = 1000

FILENAME = 'Savefiles/CarRacingDDQWeights.keras'
