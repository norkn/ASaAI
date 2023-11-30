import tensorflow as tf
from tensorflow import keras

class DeepQNet:
    
    def build_model(env, layer_sizes, activation_functions, init, learning_rate):
        state_shape = env.observation_space.shape
        action_shape = env.action_space.shape[0]
        print(action_shape)
        
        model = keras.Sequential()
        
        model.add(keras.layers.Dense(24, input_shape = state_shape, activation = 'relu', kernel_initializer = init))
        model.add(keras.layers.Dense(12, activation = 'relu', kernel_initializer = init))
        
        model.add(keras.layers.Dense(action_shape, activation = 'linear', kernel_initializer = init))
        
        model.compile(loss = tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(lr = learning_rate), metrics = ['accuracy'])
        
        return model
    
    def __init__(self, env, layer_sizes, activation_functions, init, learning_rate):
        self.model = DeepQNet.build_model(env, layer_sizes, activation_functions, init, learning_rate)
        
    def get_weights(self):
        return self.model.get_weights()
        
    def set_weights(self, weights):
        return self.model.set_weights(weights)
        
    def run(self, state):
        return self.model.predict(state)
    
    def train(self, states, qValues):
        history = self.model.fit(
            states,
            qValues,
            batch_size=64,
            epochs=2
        )
        
        return history