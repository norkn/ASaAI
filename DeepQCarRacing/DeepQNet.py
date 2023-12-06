import tensorflow as tf
from tensorflow import keras

import numpy as np

class DeepQNet:
    
    def build_model(state_shape, layer_sizes, activation_functions, init, learning_rate):
        action_shape = 5
        
        model = keras.Sequential()
        
        model.add(keras.layers.Flatten(input_shape = state_shape))
        
        model.add(keras.layers.Dense(48, activation = 'relu', kernel_initializer = init))
        model.add(keras.layers.Dense(24, activation = 'relu', kernel_initializer = init))
        
        model.add(keras.layers.Dense(action_shape, activation = 'linear', kernel_initializer = init))
        
        model.compile(loss = tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate))
        
        return model
    
    def __init__(self, state_shape, layer_sizes, activation_functions, init, learning_rate):
        self.model = DeepQNet.build_model(state_shape, layer_sizes, activation_functions, init, learning_rate)

    @staticmethod    
    def load(state_shape, epsilon_decay, path):
        dqn = DeepQNet(state_shape, [], [], None, 0)
        dqn.model = keras.models.load_model(path)
        return dqn
        
    def get_weights(self):
        return self.model.get_weights()
        
    def set_weights(self, weights):
        return self.model.set_weights(weights)
        
    def run(self, state):
        state = np.array(state)
        return self.model.predict(state.reshape(1, *state.shape)) ##################

    
    def train(self, states, qValues):        
        history = self.model.fit(
            states,
            qValues,
            batch_size = 64,
            epochs = 100,
            #verbose = 0
        )
        
        return history
