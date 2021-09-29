from lib.hybrid import Hybrid
import time
from numpy.random import seed

size = (201,151)
myseed=int(time.time())
seed(334)
print(myseed)
init_state = 185,75
real_target = 60,115

grid=Hybrid(size, real_target, init_state, plot=True, pause=1, param=0.2, tau=10, greedy=True)
print(grid.search(wait_first_obs=True))
