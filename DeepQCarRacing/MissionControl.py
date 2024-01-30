#%% Clear saved Model and RL results
from pathlib import Path
import Agent.Hyperparameters as hp

Path(hp.MODEL_PATH).unlink()
Path(hp.RL_RESULTS_PATH).unlink()

#%% Experience Replay
import ExperienceReplay

ExperienceReplay.run(num_episodes=100, sample_size=4000, load_agent=False)#True)

#%% Demo
import Demo

Demo.run(1)

#%% Compare Performance
import PerformanceComparator

PerformanceComparator.run(num_episodes=100, seed=1)

#%% Visualize
import Visualizer

Visualizer.run()
#%% Render Video
from gymnasium.wrappers import RecordVideo
