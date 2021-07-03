# Posterior Sampling Reinforcement Learning for Olfactory Search

Project for intership at DSSC

Author: Irene Brugnara
Tutor: Prof. Antonio Celani

The aim of the project is to apply Posterior Sampling Reinforcement Learning algorithm [1] for an olfactory search problem.
The model of odor detections in atmosphere is based on [2].

The code is adapted from https://github.com/IreneBrugnara/RLProject

Source files:

- `gridworld.py` contains the implementation of the class `Gridworld`, defining a two-dimensional grid environment in which the search takes place. The method `gridworld_search` implements the search algorithm based on PSRL (or a greedy algorithm if `greedy=True`);
- `animation.py` contains an example for running `gridworld_search`;
- `benchmark.py` and `analysis.py` respectively contain code to collect and process data on large-scale runs of the algorithm (the former produces a pickle file which is to be read by the latter).

[1] Osband, I., Russo, D., & Van Roy, B. (2013). (More) efficient reinforcement learning via posterior sampling. arXiv preprint arXiv:1306.0940.
[2] Celani, A., Villermaux, E., & Vergassola, M. (2014). Odor landscapes in turbulent environments. Physical Review X, 4(4), 041015.

