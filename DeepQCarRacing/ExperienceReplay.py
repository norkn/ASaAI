import numpy as np

import Agent.Hyperparameters as hp
import Main as m

def experience_replay(ddqAgent, sample_size):
    episode = ddqAgent.load_episode()
    episode = np.random.default_rng().choice(episode, size=sample_size, replace=False)

    states, q_vectors = ddqAgent.process_steps(episode)
    ddqAgent.fit(states, q_vectors)

    ddqAgent.qNet.model.save(hp.MODEL_PATH)


def run(num_episodes=100, sample_size=4000, load_agent = True):
    env, state_shape, action_shape = m.make_env(None)

    if load_agent:
        ddqAgent = m.load_agent(env, state_shape, action_shape)
    else:
        ddqAgent = m.make_agent(env, state_shape, action_shape)

    for i in range(num_episodes):
        print(f"training {i}/{num_episodes}")
        experience_replay(ddqAgent, sample_size)

