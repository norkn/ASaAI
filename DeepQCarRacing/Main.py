import gymnasium as gym

from Environment import WrappedEnv as we

def scripted_policy(state):
    action = 0
    if state[0] < 3: action = 3

    #far vision
    if state[5] == 0: action = 1 #turn right
    if state[6] == 0: action = 2 #turn left

    #near vision
    if state[3] == 0: action = 1 #turn right
    if state[4] == 0: action = 2 #turn left
    if state[0] == 0: action = 3

    return action

def nop(*args):
        return

def make_env(render_mode):
    env = gym.make("CarRacing-v2", continuous = False, render_mode = render_mode)
    env = we.WrappedEnv(env)
    
    state_shape = env.observation_space.shape    
    action_shape = (env.action_space.n, )
    
    return env, state_shape, action_shape

def main(env, num_iterations, get_action, in_loop, before_end):   
    state, info = env.reset()

    total_reward = 0
    
    for i in range(num_iterations):
        
        action = get_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
    #####<
        in_loop(state, action, reward, next_state, terminated or truncated)
    ########>
        state = next_state
        
        if terminated or truncated:
            state, info = env.reset()

        total_reward +=  reward

    env.close()
    ##<
    before_end()
    ###>
    avg_reward_per_100 = 100 * (total_reward / num_iterations)
    return avg_reward_per_100
