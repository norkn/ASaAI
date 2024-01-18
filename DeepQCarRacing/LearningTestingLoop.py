#%%
import ReinforcementLearning_epsilon as rl
import PerformanceComparator as pc

SESSIONS = 1
LEARNING_SESSIONS = 1

for i in range(SESSIONS):
    pc.run()
    
    for i in range(LEARNING_SESSIONS):
        rl.run()
        print("weights saved")
    
#%%
import numpy as np

rl_results = np.load('Savefiles/rl_results.npy',   mmap_mode="r")
scripted_results = np.load('Savefiles/scripted_results.npy',   mmap_mode="r")

for r in zip(scripted_results, rl_results):
    print(r)

# %%
