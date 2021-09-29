from lib.thompson import Thompson
import time
from numpy.random import seed

size = 201,151
myseed=int(time.time())
seed(9998)
print(myseed)
init_state = 195,75
real_target = 60,115

grid=Thompson(size, real_target, init_state, plot=True, pause=1, tau=50, greedy=False, prob=0.8)
t=grid.search(wait_first_obs=True)
print(t)
