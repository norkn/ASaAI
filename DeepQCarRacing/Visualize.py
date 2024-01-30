# %%
import matplotlib.pyplot as plt
import numpy as np

rl_results = np.load('Savefiles/rl_results.npy',   mmap_mode="r")
scripted_results = np.load('Savefiles/scripted_results.npy',   mmap_mode="r")

len_r = min(len(rl_results), len(scripted_results))

rl_results = rl_results[:len_r]
scripted_results = scripted_results[:len_r]

print(f"rl avg: {np.average(rl_results)}")#np.sum(rl_results) / len(rl_results)}")
print(f"scripted avg: {np.average(scripted_results)}")#np.sum(scripted_results) / len(scripted_results)}")

results_diff = rl_results - scripted_results

plt.style.use('_mpl-gallery')

x = 0. + np.arange(len_r)

fig, ax = plt.subplots(figsize=(16,4))

ax.plot(x, rl_results, 'r')
ax.plot(x, scripted_results, 'g')
ax.plot(x, results_diff, 'b')

ax.set(xlim=(0, len_r),
       ylim=(0, 1000))

plt.show()

#%%
import numpy as np
rl_results = np.load('Savefiles/rl_results.npy',   mmap_mode="r")
scripted_results = np.load('Savefiles/scripted_results.npy',   mmap_mode="r")
len_r = min(len(rl_results), len(scripted_results))

for r in zip(scripted_results, rl_results):
       print(r)