import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import cosh
from common import Gridworld


class Thompson(Gridworld):
    def __init__(self, nrows, ncols, source, init_state, plot=True, pause=0, param=0.2):
        super().__init__(nrows, ncols, source, init_state, plot, pause, param)
        

    #thompson: thompson sampling to choose a new estimate for the target position

    def thompson(self):
        index=np.random.choice(self.dimensions[0]*self.dimensions[1], p=self.belief.ravel())
        self.estimated_target = np.unravel_index(index, self.dimensions)



#gridworld_search: simulates Thompson or greedy algorithm and returns the number of steps t taken until target is reached

def thompson_search(grid, tau, greedy=False, maxiter=np.inf, wait_first_obs=True):
    obs=0
    if wait_first_obs:  # stay still until first observation
        while obs==0:
            obs=grid.observe()
        grid.update_efficient(obs)    # INDENT
    else:
        grid.first_update()
    t=0
    while not grid.done:
        if greedy:
            grid.greedy()
        else:
            grid.thompson()
        i=0
        reached_est_target=False
        while not reached_est_target and i < tau and not grid.done:
            if grid.plot:
                grid.show(t, obs)
            action=grid.chase()
            grid.advance(action)
            obs=grid.observe()
            grid.update_efficient(obs)
            t+=1
            i+=1
            if t>=maxiter:
               return t
            reached_est_target = grid.state == grid.estimated_target
    if grid.plot:
        grid.show(t, obs)
    return t
