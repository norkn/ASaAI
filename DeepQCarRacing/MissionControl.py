#%% Clear saved Model and RL results

#%% Experience Replay
import ExperienceReplay

ExperienceReplay.run(num_episodes=100, sample_size=4000, load_agent=False)#True)

#%% Demo
import Demo

Demo.run(1)

#%% Compare Performance
import PerformanceComparator

PerformanceComparator.run(200)

#%% Visualize
import Visualize

Visualize.run()
#%% Render Video
