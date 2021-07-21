import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import cosh
from common import Gridworld


class Thompson(Gridworld):
    def __init__(self, nrows, ncols, source, init_state, render=True, pause=0, param=0.2):
        super().__init__(nrows, ncols, source, init_state, render, pause, param)
        self.estimated_target=(None,None)     # current estimate of target position

        if self.render:
          self.scat_target = self.ax.scatter(self.estimated_target[1], self.estimated_target[0], color='b', marker='x', label="estimated target", s=150)

    #show: display a graphical representation of the grid

    def update_plots(self, t, obs):
        super().update_plots(t, obs)
        self.scat_target.set_offsets([self.estimated_target[1], self.estimated_target[0]])
        

        


    #advance: apply action and transition to the new state

    def advance(self, action):
        self.state = self.move(action)
        self.done = (self.state==self.source)

    #thompson: thompson sampling to choose a new estimate for the target position
    #greedy: greedy choice for the next estimated target position

    def thompson(self):
        index=np.random.choice(self.dimensions[0]*self.dimensions[1], p=self.belief.ravel())
        self.estimated_target = np.unravel_index(index, self.dimensions)
        
    def greedy(self):
        index=np.random.choice(np.flatnonzero(self.belief == self.belief.max()))  # in case of tie (the belief has multiple maxima) I choose randomly (otherwise "index=np.argmax(self.belief)" would pick always the first element and this introduces a bias)
        self.estimated_target = np.unravel_index(index, self.dimensions)
        #if np.count_nonzero(self.belief == self.belief.max())>1:
        #    print("tie", np.count_nonzero(self.belief == self.belief.max()))       
        
         

    #policy: pick action to get one step closer to the current estimate of the target

    def policy(self):
        var_r=self.estimated_target[0]-self.state[0]
        var_c=self.estimated_target[1]-self.state[1]
        action_r=np.sign(var_r) # direction of movement along horizontal axis
        action_c=np.sign(var_c) # direction of movement along vertical axis
        if action_r==0 or action_c==0: # stay still on at least one axis
            return (action_r, action_c)
        elif np.random.uniform()>0.5: # if two actions are equivalent, choose randomly
            return (action_r, 0)
        else:
            return (0, action_c)
            



#gridworld_search: simulates Thompson or greedy algorithm and returns the number of steps t taken until target is reached

def gridworld_search(grid, tau, greedy=False, maxiter=np.inf, wait_first_obs=True):
    obs=0
    if wait_first_obs:
        while obs==0:
            obs=grid.observe()
        grid.update_efficient(obs)    # INDENT
    t=0
    while not grid.done:
        if greedy:
            grid.greedy()
        else:
            grid.thompson()
        i=0
        reached_est_target=False
        while not reached_est_target and i < tau and not grid.done:
            if grid.render:
                grid.show(t, obs)
            action=grid.policy()
            grid.advance(action)
            obs=grid.observe()
            grid.update_efficient(obs)
            t+=1
            i+=1
            if t>=maxiter:
               return t
            reached_est_target = grid.state == grid.estimated_target
    if grid.render:
        grid.show(t, obs)
    return t
