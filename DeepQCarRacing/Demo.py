import Agent.Hyperparameters as hp

import Main as m

def run():
    
    env, state_shape, action_shape = m.make_env("human")
    
    ddqAgent = m.load_agent(env, state_shape, action_shape)

    total = 0

    def in_loop(s, a, r, n, d):
      nonlocal total
      total += r

    def end_episode():
      nonlocal total
      print("episode ends. total: ", total)
      total = 0

    result = m.main(env, hp.RUNNING_NUM_EPISODES, ddqAgent.get_action, in_loop, end_episode)
    print(f"avg reward per episode: {result}")
    
run()
