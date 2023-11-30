import tensorflow as tf
from tensorflow import keras
class DeepQNet:
    
    def build_model(env, layer_sizes, activation_functions, init, learning_rate):
        state_shape = env.observation_space.shape
        action_shape = 5
        
        model = keras.Sequential()
        
        model.add(keras.layers.Flatten(input_shape = (96, 96, 3)))#state_shape))
        #model.add(keras.layers.Input(shape = state_shape))
        #model.add(keras.layers.Flatten())
        
        model.add(keras.layers.Dense(24, activation = 'relu', kernel_initializer = init))
        model.add(keras.layers.Dense(12, activation = 'relu', kernel_initializer = init))
        
        model.add(keras.layers.Dense(action_shape, activation = 'linear', kernel_initializer = init))
        
        model.compile(loss = tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(lr = learning_rate), metrics = ['accuracy'])
        
        return model
    
    def __init__(self, env, layer_sizes, activation_functions, init, learning_rate):
        self.model = DeepQNet.build_model(env, layer_sizes, activation_functions, init, learning_rate)
        
    def __init__(self, path):
        self.model = keras.models.load_model(path)
        
    def get_weights(self):
        return self.model.get_weights()
        
    def set_weights(self, weights):
        return self.model.set_weights(weights)
        
    def run(self, state):
        return self.model.predict(state.reshape(1, *state.shape))

    
    def train(self, states, qValues):        
        history = self.model.fit(
            states,
            qValues,
            batch_size=64,
            epochs=100
        )
        
        return history