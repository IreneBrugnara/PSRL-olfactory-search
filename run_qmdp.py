from lib.qmdp import Qmdp
import time
from numpy.random import seed

size = 201,151
myseed=int(time.time())
seed(myseed)
print(myseed)
init_state = 185,75
real_target = 60,115
grid=Qmdp(size, real_target, init_state, plot=True, pause=0.0, param=0.2, gamma=0.9)
print(grid.search(wait_first_obs=True, max_iter=313))

