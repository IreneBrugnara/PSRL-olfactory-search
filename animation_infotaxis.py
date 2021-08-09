#from gridworld_infotaxis_modificato import Gridworld
from time import time
import numpy as np
from infotaxis import *
   
nrows=201
ncols=151
myseed=int(time())
np.random.seed(myseed) 
print(myseed)
init_state = 185,75
source = 60,115
grid=Infotaxis(nrows, ncols, source, init_state, plot=True, pause=0, param=0.2)
print(infotaxis_search(grid, threshold=2, wait_first_obs=True, maxiter=2000))

