from gridworld import *
import time

nrows=200
ncols=100
myseed=3  
np.random.seed(myseed)  
print(myseed)
init_state = 150,30
real_target = 20,60
grid=Gridworld(nrows, ncols, real_target, init_state, render=True)
print(gridworld_search(grid, 10, greedy=False))
