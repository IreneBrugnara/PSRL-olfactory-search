import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import cosh


class Gridworld():
    def __init__(self, nrows, ncols, real_target, init_state, render=True, param=0.2):
        self.dimensions=(nrows,ncols)    # dimensions of the grid
        self.state=init_state    # current state
        self.belief=np.ones(self.dimensions)/(nrows*ncols)    # uniform prior
        self.param=param    # parameter v/U in the observation model
        self.done=(real_target==init_state)    # flag set to true when real target is reached (reward=1)
        self.real_target=real_target    # real target position
        self.estimated_target=(None,None)     # current estimate of target position
        self.render=render    # if true, a graphical representation of the gridworld is printed

        if self.render:
          self.fig = plt.figure(figsize=(16, 10))
          self.ax = self.fig.add_subplot(111)
          self.im = self.ax.imshow(self.belief, cmap='Greens')
          self.ax.set_xticks(np.arange(self.dimensions[1], dtype=np.int))
          self.ax.set_yticks(np.arange(self.dimensions[0], dtype=np.int))
          self.scat_real_target = self.ax.scatter(self.real_target[1], self.real_target[0], color='r', marker='*', label="real target", s=150)
          self.scat_mel = self.ax.scatter(self.state[1], self.state[0], color='orange', marker='o', label="current position", zorder=1, s=150)
          self.scat_target = self.ax.scatter(self.estimated_target[1], self.estimated_target[0], color='b', marker='x', label="estimated target", s=150)
          self.fig.colorbar(self.im, ax=self.ax, ticks=None)
          plt.show(block=False)

    #show: display a graphical representation of the grid

    def show(self, t):
        self.im.autoscale()
        self.im.set_array(self.belief)
        self.scat_me = self.ax.scatter(self.state[1], self.state[0], color='y', marker='o', s=16, zorder=0.1, label="visited positions")
        self.scat_mel.set_offsets([self.state[1], self.state[0]])
        self.scat_target.set_offsets([self.estimated_target[1], self.estimated_target[0]])
        self.ax.set_title("t="+str(t))
        self.fig.canvas.draw()
        if t==0:
            self.ax.legend(bbox_to_anchor=(-0.2, 0.5))
        plt.pause(0.01)
        


    #update: posterior calculation given an observation

    def update(self, observation):
        nrows, ncols=self.dimensions
        likelihood_matrix=np.empty_like(self.belief)
        '''
        for i in range(nrows):
            for j in range(ncols):
                likelihood_matrix[i,j]=self.likelihood(observation, (i,j))
        '''
        for pos in product(range(nrows), range(ncols)):    # loop over each state in the grid
            likelihood_matrix[pos]=self.likelihood(observation, pos)    # compute likelihood
        self.belief=np.multiply(self.belief, likelihood_matrix)
        self.belief/=np.sum(self.belief)     # normalize posterior
        

    #step: apply action and transition to the new state

    def step(self, action):
        new_state=self.state[0]+action[0], self.state[1]+action[1]
        #if the new position is outside the grid the state is not updated
        if new_state[0]<self.dimensions[0] and new_state[0]>=0 and new_state[1]<self.dimensions[1] and new_state[1]>=0:
            self.state=new_state
        self.done = (self.state==self.real_target)

    #thompson: thompson sampling to choose a new estimate for the target position
    #greedy: greedy choice for the next estimated target position

    def thompson(self):
        index=np.random.choice(range(self.dimensions[0]*self.dimensions[1]), p=self.belief.flatten())
        self.estimated_target = np.unravel_index(index, self.dimensions)
    def greedy(self):
        index=np.argmax(self.belief)
        self.estimated_target = np.unravel_index(index, self.dimensions)

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
            
    #bernoulli_param: compute the parameter of the Bernoulli for the model of observations
            
    def bernoulli_param(self, state, source):
        x=state[0]-source[0]
        y=state[1]-source[1]
        if x<=0:
            p=0   # no observations can be made behind the source
        else:
            if abs(y/x/self.param) < 350:    # to prevent overflow
                p=1/(cosh(y/x/self.param))**2
            else:
                p=0
        return p

    #likelihood: compute the likelihood of a given observation y assuming that the target is est_target

    def likelihood(self, y, est_target):
        p=self.bernoulli_param(self.state, est_target)
        if self.state==est_target:     # this is because the source is a terminal state, so 
            return 0                   # the state just visited was not certainly the source regardless of y
        return p if y==1 else 1-p

    #observe: draw a real (random) observation from the environment

    def observe(self):
        p=self.bernoulli_param(self.state, self.real_target)
        return 1 if np.random.uniform()<p else 0
        


#gridworld_search: simulates Thompson or greedy algorithm and returns the number of steps t taken until target is reached

def gridworld_search(grid, tau, greedy=False):
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
                grid.show(t)
            action=grid.policy()
            grid.step(action)
            obs=grid.observe()
            grid.update(obs)
            t+=1
            i+=1
            reached_est_target = grid.state == grid.estimated_target
    if grid.render:
        grid.show(t)
    return t
