import tensorflow as tf

import DeepQNet

#training
#experience replay
#set target net weights to Q net weights

class DoubleDeepQAgent:
    
    def __init__(self, env, layer_sizes, activation_functions, init, learning_rate):
        self.qNet = DeepQNet(env, layer_sizes, activation_functions, init, learning_rate)
        self.targetNet = DeepQNet(env, layer_sizes, activation_functions, init, learning_rate)