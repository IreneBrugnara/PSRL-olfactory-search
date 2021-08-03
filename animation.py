from thompson import *
import time

nrows=201#301#
ncols=151#225#
myseed=int(time.time())
np.random.seed(12345)
print(myseed)
init_state = 185,75#277,112#
real_target = 60,115#90,172#

grid=Thompson(nrows, ncols, real_target, init_state, plot=True, pause=0, param=0.2)
grid.field()
print(thompson_search(grid, 15, greedy=True, wait_first_obs=True))

