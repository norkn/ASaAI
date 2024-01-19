import Hyperparameters as hp
import Main as m

def train_on_old_data():
    env, state_shape, action_shape = m.make_env(None)
    ddqAgent = m.load_agent(env, state_shape, action_shape)

    training_states, training_q_vectors = ddqAgent.load_and_process_episode()
    ddqAgent.train_offline(training_states, training_q_vectors)

    ddqAgent.qNet.model.save(hp.FILENAME)

