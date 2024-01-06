import Agent.Hyperparameters as hp

import Main as m

env, state_shape, action_shape = m.make_env(None)

ddqAgent = m.make_agent(env, state_shape, action_shape)

# ddqAgent = m.load_agent(env, state_shape, action_shape)

ddqAgent.train_on_saved_data()

ddqAgent.qNet.model.save(hp.FILENAME)
print('weights saved, trained on saved data')
