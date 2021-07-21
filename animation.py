from gridworld import *
import time

nrows=201
ncols=201
myseed=712
np.random.seed(myseed)  
print(myseed)
init_state = 150,100
real_target = 25,60
grid=Thompson(nrows, ncols, real_target, init_state, render=True, pause=2)
print(gridworld_search(grid, 1, greedy=False, wait_first_obs=False))


