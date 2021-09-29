from lib.action_voting import ActionVoting
import time
from numpy.random import seed


size = 201,151
myseed=int(time.time())
seed(122)
print(myseed)
init_state = 185,75
real_target = 60,115
grid=ActionVoting(size, real_target, init_state, plot=True, pause=2, param=0.2, prob=0.7)
print(grid.search(wait_first_obs=True, max_iter=1500))

