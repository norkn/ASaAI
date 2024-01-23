import Agent.Hyperparameters as hp
import Main as m

import numpy as np

def train_on_old_data():
    env, state_shape, action_shape = m.make_env(None)
    ddqAgent = m.load_agent(env, state_shape, action_shape)

    training_states, training_q_vectors = ddqAgent.load_and_process_episode()
    
    # sq = zip(training_states, training_q_vectors)
    # psq = np.random.permutation(sq)

    # s = [e[0] for e in psq]
    # q = [e[1] for e in psq]
    
    ddqAgent.fit(training_states, training_q_vectors)

    ddqAgent.qNet.model.save(hp.FILENAME)


for i in range(2):
    train_on_old_data()
