LAYER_SIZES = [128, 64]
LAYER_ACTIVATIONS = 3*['linear']#['relu', 'relu', 'linear']
LEARNING_RATE = 0.1#0.005

EPOCHS = 100

TRAINING_ITERATION = 1#000
WEIGHTS_TRANSFER_ITERATION = 1#500

#one step is equivalent to about 0.029s in human experience, ie 100 steps is about 3s
GAMMA = 0.95#8 #rewards 50 steps in the future get weighted at 36.4%, 100 steps in the future 13.2%
EPSILON_DECAY = 0.99#8 #after 1000 steps 13.5% chance of taking a random action

EPISODES_TRAINING = 50
EPISODES_RUNNING = 1000

FILENAME = 'CarRacingDDQWeights.keras'
