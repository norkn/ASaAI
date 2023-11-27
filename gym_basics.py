import gymnasium as gym



#env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
#env = gym.make("CarRacing-v2", render_mode="human")
observation, info = env.reset()



for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
