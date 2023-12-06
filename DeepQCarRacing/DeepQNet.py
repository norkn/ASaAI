import tensorflow as tf
from tensorflow import keras

import numpy as np

class DeepQNet:
    
    def build_model(state_shape, action_shape, layer_sizes, activation_functions, init, learning_rate):
        
        model = keras.Sequential()
        
        model.add(keras.layers.Flatten(input_shape = state_shape))
        
        model.add(keras.layers.Dense(layer_sizes[0], activation = activation_functions[0], kernel_initializer = init))
        model.add(keras.layers.Dense(layer_sizes[1], activation = activation_functions[1], kernel_initializer = init))
        
        model.add(keras.layers.Dense(action_shape[0], activation = activation_functions[-1], kernel_initializer = init))
        
        model.compile(loss = tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate))
        
        return model
    
    def __init__(self, state_shape, action_shape, layer_sizes, activation_functions, init, learning_rate):
        if state_shape == None:
            self.model = None
        else:
            self.model = DeepQNet.build_model(state_shape, action_shape, layer_sizes, activation_functions, init, learning_rate)

    @staticmethod    
    def load(path):
        dqn = DeepQNet(*[None]*6)
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
            batch_size = 100,
            epochs = 1000,
            #verbose = 0
        )
        
        return history
