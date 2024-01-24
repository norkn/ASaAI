import Agent.Hyperparameters as hp
import Main as m

import numpy as np

def train_on_old_data():
    env, state_shape, action_shape = m.make_env(None)
    ddqAgent = m.load_agent(env, state_shape, action_shape)

    training_states, training_q_vectors = ddqAgent.load_and_process_episode()
       
    ddqAgent.fit(training_states, training_q_vectors)

    ddqAgent.qNet.model.save(hp.FILENAME)


for i in range(100):
    train_on_old_data()
