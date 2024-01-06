import numpy as np

import ReinforcementLearning as rl
import PerformanceComparator as pc

SESSIONS = 10
LEARNING_SESSIONS = 1

for i in range(SESSIONS):
    for i in range(LEARNING_SESSIONS):
        rl.run()
        print("weights saved")

    pc.run()
    
#%%
import numpy as np

rl_results = np.load('Savefiles/rl_results.npy',   mmap_mode="r")
scripted_results = np.load('Savefiles/scripted_results.npy',   mmap_mode="r")

for r in zip(scripted_results, rl_results):
    print(r)