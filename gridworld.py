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
          self.scat_real_target = self.ax.scatter(self.real_target[1], self.real_target[0], color='r', marker='*', label="real target", s=150)
          self.scat_me = self.ax.scatter(self.state[1], self.state[0], color='y', marker='o', s=5, zorder=0.1, label="visited positions")
          self.scat_mel = self.ax.scatter(self.state[1], self.state[0], color='orange', marker='o', label="current position", zorder=1, s=100)
          self.scat_target = self.ax.scatter(self.estimated_target[1], self.estimated_target[0], color='b', marker='x', label="estimated target", s=150)
          self.scat_obs = self.ax.scatter(None, None, color='purple', marker='o', label="observations", s=5, zorder=0.2)
          self.fig.colorbar(self.im, ax=self.ax, ticks=None)
          self.text = self.fig.text(0.4, 0.6, "")
          plt.show(block=False)

    #show: display a graphical representation of the grid

    def show(self, t, obs):
        self.im.autoscale()
        self.im.set_array(self.belief)
        self.scat_me.set_offsets(np.vstack([self.scat_me.get_offsets(), np.array([self.state[1],self.state[0]])]))
        self.scat_mel.set_offsets([self.state[1], self.state[0]])
        self.scat_target.set_offsets([self.estimated_target[1], self.estimated_target[0]])
        if obs:
            self.scat_obs.set_offsets(np.vstack([self.scat_obs.get_offsets(), np.array([self.state[1],self.state[0]])]))
            
        self.ax.set_title("t="+str(t))
        if t==0:
            self.ax.legend(bbox_to_anchor=(-0.2, 0.5))
        self.text.set_text("y=1" if obs else "")
        
        self.fig.canvas.draw()
        #plt.pause(0.1)     # comment this to have fastest animation
        
    #field: display the model of observations
        
    def field(self):
        field=np.empty_like(self.belief)
        nrows, ncols=self.dimensions
        for pos in product(range(nrows), range(ncols)):    # loop over each state in the grid
            field[pos] = self.bernoulli_param(pos, self.real_target)
        plt.imshow(field)
        plt.colorbar()
        plt.show()

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
        
    #update_efficient: much faster (but less readable) version of update(), that vectorizes the computation of the likelihood with Numpy
        
    def update_efficient(self, observation):
        nrows, ncols=self.dimensions
        x = np.tile(np.arange(self.state[0]).reshape(self.state[0],1), (1,ncols))
        y = np.tile(np.arange(ncols), (self.state[0],1))
        x = self.state[0] - x     # non inplace
        y = self.state[1] - y
        #x *= -1
        #x += self.state[0]
        #y *= -1
        #y += self.state[1]
        np.seterr(over='ignore')     # ignore overflow errors in the computation of p
        p=1/(np.cosh(y/x/self.param))**2
        np.seterr(over='warn')
        if observation==0:
            p=1-p    # non inplace
            #p *= -1
            #p += 1
        
        if observation==0:
            lkh=np.ones((nrows,ncols))
        else:
            lkh=np.zeros((nrows,ncols))
        lkh[:self.state[0]] = p
        lkh[self.state] = 0
        
        self.belief = self.belief * lkh
        self.belief/=np.sum(self.belief)

        

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
        index=np.random.choice(self.dimensions[0]*self.dimensions[1], p=self.belief.ravel())
        self.estimated_target = np.unravel_index(index, self.dimensions)
    def greedy(self):
        if np.all(self.belief==self.belief[0,0]): # at the first step the belief is flat so I choose randomly (otherwise np.argmax would pick always the first element and this introduces a bias)
            index=np.random.randint(self.dimensions[0]*self.dimensions[1])
        else:
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
    obs=0
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
            grid.step(action)
            obs=grid.observe()
            grid.update_efficient(obs)
            t+=1
            i+=1
            reached_est_target = grid.state == grid.estimated_target
    if grid.render:
        grid.show(t, obs)
    return t
