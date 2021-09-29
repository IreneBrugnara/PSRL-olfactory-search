from lib.infotaxis import Infotaxis
from lib.dualmode import DualMode
import time
from numpy.random import seed

size = 201, 151
myseed=int(time.time())
seed(45)
print(myseed)
init_state = 185,75
real_target = 60,115

#grid=Infotaxis(size, real_target, init_state, plot=True, pause=1, param=0.2)
#print(grid.search(wait_first_obs=True))

grid=DualMode(size, real_target, init_state, plot=True, pause=0.1, param=0.2, threshold=6)
print(grid.search(max_iter=1000, wait_first_obs=False))
