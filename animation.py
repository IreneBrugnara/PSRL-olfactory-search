from gridworld import *
import time

nrows=20
ncols=10
myseed=3  
np.random.seed(myseed)  
print(myseed)
init_state = 15,3
real_target = 2,6 
grid=Gridworld(nrows, ncols, real_target, init_state, render=True)
print(gridworld_search(grid, 10, greedy=False))
