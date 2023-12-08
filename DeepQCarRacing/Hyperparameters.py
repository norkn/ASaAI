LAYER_SIZES = [48, 24]
LAYER_ACTIVATIONS = ['relu', 'relu', 'linear']
LEARNING_RATE = 0.005

#one step is equivalent to about 0.029s in human experience, ie 100 steps is about 3s
GAMMA = 0.98 #rewards 50 steps in the future get weighted at 36.4%, 100 steps in the future 13.2%
EPSILON_DECAY = 0.998 #after 1000 steps 13.5% chance of taking a random action

ITERATIONS_TRAINING = 50
ITERATIONS_RUNNING = 100

FILENAME = 'CarRacingDDQWeights.keras'
