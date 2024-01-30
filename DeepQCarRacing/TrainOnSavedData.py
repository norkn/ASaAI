import Agent.Hyperparameters as hp

import Main as m

env, state_shape, action_shape = m.make_env(None)

ddqAgent = m.make_agent(env, state_shape, action_shape)
# ddqAgent = m.load_agent(env, state_shape, action_shape)

ddqAgent.fit_to_measured_q_values()

ddqAgent.qNet.model.save(hp.MODEL_PATH)
print('weights saved, trained on saved data')
