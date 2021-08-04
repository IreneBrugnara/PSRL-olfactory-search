import numpy as np
import matplotlib.pyplot as plt
from common import Gridworld
from time import time



#infotaxis_search: simulates Infotaxis and returns the number of steps t taken until target is reached; with threshold>0 implements Dual Mode control (under entropy threshold, the greedy action is taken)

def infotaxis_search(grid, threshold=0, maxiter=np.inf, wait_first_obs=True):
    obs=0
    if wait_first_obs:
        while obs==0:
            obs=grid.observe()
        grid.update_efficient(obs)    # INDENT
    else:
        grid.first_update()
    t=0
    if grid.plot:
         grid.show(t, obs)
    possible_actions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]  # set of candidate actions (still, right, left, down, up)
    while not grid.done and t<=maxiter:
        t+=1    # andrebbe incrementato dopo lo step, ma è per il plot
        if grid.entropy>=threshold:
            obs=grid.infotaxis_step(possible_actions)
        else:
            obs=grid.mls_step()
        if grid.plot:
            grid.show(t, obs)
    return t
    
    
   
nrows=201
ncols=151
myseed=int(time())
np.random.seed(1627811999) 
print(myseed)
init_state = 185,75
source = 60,115
grid=Gridworld(nrows, ncols, source, init_state, plot=True, pause=0, param=0.2)
print(infotaxis_search(grid, threshold=2, wait_first_obs=True, maxiter=7000))

