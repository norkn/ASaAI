import Agent.DoubleDeepQAgent as ddqa
import Agent.Hyperparameters as hp

import Main as m

def run():
    
    env, state_shape, action_shape = m.make_env("human")
    
    ddqAgent = ddqa.DoubleDeepQAgent(env, 
                                     state_shape,
                                     action_shape,
                                     hp.LAYER_SIZES, 
                                     hp.LAYER_ACTIVATIONS, 
                                     None, 
                                     hp.LEARNING_RATE,
                                     hp.LOSS,
                                     hp.OPTIMIZER,
                                     hp.NUM_BATCHES,
                                     hp.EPOCHS,
                                     hp.SAMPLE_SIZE, 
                                     hp.GAMMA, 
                                     hp.EPSILON_DECAY)

    m.main(env, hp.TRAINING_STEPS, m.scripted_policy, ddqAgent.record_training_data, ddqAgent.process_and_save_training_data)
    
    
run()
