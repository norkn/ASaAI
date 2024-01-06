import Agent.DoubleDeepQAgent as ddqa
import Agent.Hyperparameters as hp

import Main as m

def run():
    
    env, state_shape, action_shape = m.make_env("human")
    
    ddqAgent = ddqa.DoubleDeepQAgent.load(env,
                                          state_shape,
                                          action_shape,
                                          hp.FILENAME,
                                          hp.NUM_BATCHES,
                                          hp.EPOCHS,
                                          hp.SAMPLE_SIZE, 
                                          hp.GAMMA, 
                                          hp.EPSILON_DECAY)
    
    def in_loop(state, action, reward, next_state, done):
        ddqAgent.record_training_data(state, action, reward, next_state, done)
        print(f"action {action}")

        if in_loop.i % hp.SAMPLE_SIZE == 0:
            ddqAgent.train_on_new_data()
        
        in_loop.i += 1
    in_loop.i = 1
    
    before_end = lambda: ddqAgent.qNet.model.save(hp.FILENAME)

    m.main(env, hp.TRAINING_STEPS, ddqAgent.get_action_by_distribution, in_loop, before_end)
    
    
run()
