import Agent.Hyperparameters as hp

import Main as m

def run():
    
    env, state_shape, action_shape = m.make_env("human")
    
    ddqAgent = m.load_agent(env, state_shape, action_shape)

    total = 0

    def in_loop(s, a, r, n, d):
      nonlocal total
      total += r

      ddqAgent.record_episode(s, a, r, n, d)


    def end_episode():
      nonlocal total, ddqAgent

      print("episode ends. total: ", total)
      total = 0

      ddqAgent.save_episode()
      ddqAgent.reset_episode()

    result = m.main(env, hp.RUNNING_NUM_EPISODES, ddqAgent.get_action, in_loop, end_episode)
    print(f"avg reward per episode: {result}")
    
run()
