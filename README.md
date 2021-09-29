# Posterior Sampling Reinforcement Learning for Olfactory Search

Project for intership at DSSC

Author: Irene Brugnara  
Tutor: Prof. Antonio Celani

The aim of the project is to apply Posterior Sampling Reinforcement Learning algorithm [1] for an olfactory search problem.  
The model of odor detections in atmosphere is based on [2].

The code is adapted from https://github.com/IreneBrugnara/RLProject

Source files:

- The file `lib/gridworld.py` contains the implementation of the abstract base class `Gridworld`, defining a two-dimensional grid environment in which the search takes place. The method `search` implements the search algorithm. Classes defined in all other files in `lib` derive from `Gridworld`.
- The directory `animation` contains classes for visualizing search trajectories.
- Files `benchmark_1.py` and `analysis_1.py` contain code to respectively collect and process data on large-scale runs of the algorithm (the former produces a pickle file which is to be read by the latter). The benchmark can be run in parallel with multiple cores. Similarly for `benchmark_2.py` and `analysis_2.py`.

[1] Osband, I., Russo, D., & Van Roy, B. (2013). (More) efficient reinforcement learning via posterior sampling. arXiv preprint arXiv:1306.0940.  
[2] Celani, A., Villermaux, E., & Vergassola, M. (2014). Odor landscapes in turbulent environments. Physical Review X, 4(4), 041015.

