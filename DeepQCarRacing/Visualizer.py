# %%
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

import Agent.Hyperparameters as hp

def plot_track_sequence(scripted_results, rl_results):
       len_r = min(len(rl_results), len(scripted_results))

       rl_results = rl_results[:len_r]
       scripted_results = scripted_results[:len_r]

       results_diff = rl_results - scripted_results

       plt.style.use('_mpl-gallery')

       x = 0. + np.arange(len_r)

       fig, ax = plt.subplots(figsize=(16,4))

       ax.plot(x, rl_results, 'r')
       ax.plot(x, scripted_results, 'g')
       # ax.plot(x, results_diff, 'b')

       ax.set(xlim=(0, len_r),
              ylim=(0, 1000))

       plt.show()

def plot_histogram(results):
       BAR_WIDTH = 100
       plt.style.use('_mpl-gallery')

       len_r = len(results)

       histogram = defaultdict(lambda: 0)
       for result in results:
              key = int(result / BAR_WIDTH) * BAR_WIDTH
              histogram[key] += 1

       x = np.array(sorted(histogram.keys()))
       y = np.array([histogram[key] for key in x])

       x_lim = (-100, 1000)
       y_lim = (0, 50)#np.max(y) + 1)

       fig, ax = plt.subplots(figsize=(16,4))

       ax.bar(x, y, width=BAR_WIDTH)

       ax.set(xlim=x_lim,
              yticks = np.arange(*y_lim, 10))

       print("Sample size: ", len_r)
       plt.show()

def run():
       rl_results = np.load(hp.RL_RESULTS_PATH,   mmap_mode="r")
       scripted_results = np.load(hp.SCRIPTED_RESULTS_PATH,   mmap_mode="r")

       rl_avg = np.average(rl_results)
       scripted_avg = np.average(scripted_results)

       print("Comparison: Red RL, Green Scripted")
       plot_track_sequence(scripted_results, rl_results)

       print("Scripted results:      avg:", scripted_avg)
       plot_histogram(scripted_results)
       print("RL results:      avg:", rl_avg)
       plot_histogram(rl_results)

