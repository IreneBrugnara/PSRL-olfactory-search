import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import cosh


class Gridworld():
    def __init__(self, nrows, ncols, source, init_state, render=True, pause=0, param=0.2):
        self.dimensions=(nrows,ncols)    # dimensions of the grid
        self.state=init_state    # current state
        self.belief=np.ones(self.dimensions)/(nrows*ncols)    # uniform prior
        self.param=param    # parameter v/U in the observation model
        self.done=(source==init_state)    # flag set to true when real target is reached (reward=1)
        self.source=source    # real target position
        self.render=render    # if true, a graphical representation of the gridworld is printed
        self.pause=pause    # time interval between animation frames

        if self.render:
          self.fig = plt.figure(figsize=(16, 10))
          self.ax = self.fig.add_subplot(111)
          self.im = self.ax.imshow(self.belief, cmap='Greens')
          self.scat_source = self.ax.scatter(self.source[1], self.source[0], color='r', marker='*', label="real target", s=150)
          self.scat_me = self.ax.scatter(self.state[1], self.state[0], color='y', marker='o', s=5, zorder=0.1, label="visited positions")
          self.scat_mel = self.ax.scatter(self.state[1], self.state[0], color='orange', marker='o', label="current position", zorder=1, s=100)
          self.scat_obs = self.ax.scatter(None, None, color='purple', marker='o', label="observations", s=5, zorder=0.2)
          self.fig.colorbar(self.im, ax=self.ax, ticks=None)
          self.text = self.fig.text(0.4, 0.6, "")
          plt.show(block=False)

    def update_plots(self, t, obs):
        self.im.autoscale()
        self.im.set_array(self.belief)
        self.scat_me.set_offsets(np.vstack([self.scat_me.get_offsets(), np.array([self.state[1],self.state[0]])]))
        self.scat_mel.set_offsets([self.state[1], self.state[0]])
        if obs:
            self.scat_obs.set_offsets(np.vstack([self.scat_obs.get_offsets(), np.array([self.state[1],self.state[0]])]))
            
        self.ax.set_title("t="+str(t))
        if t==0:
            self.ax.legend(bbox_to_anchor=(-0.2, 0.5))
        self.text.set_text("y=1" if obs else "")
    
    
    #show: display a graphical representation of the grid

    def show(self, t, obs):
        self.update_plots(t, obs)        
        self.fig.canvas.draw()
        if self.pause!=0:     # pause=0 is the fastest animation
            plt.pause(self.pause)
        
    #field: display the model of observations
        
    def field(self):
        field=np.empty_like(self.belief)
        nrows, ncols=self.dimensions
        for pos in product(range(nrows), range(ncols)):    # loop over each state in the grid
            field[pos] = self.bernoulli_param(pos, self.source)
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
        self.belief = self.belief * self.likelihood_vectorial(observation, self.state)
        self.belief/=np.sum(self.belief)

        
            
    
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

    
    def bernoulli_param_vectorial(self, state):
        nrows, ncols=self.dimensions
        x = np.tile(np.arange(state[0]).reshape(state[0],1), (1,ncols))
        y = np.tile(np.arange(ncols), (state[0],1))
        x = state[0] - x
        y = state[1] - y
        np.seterr(over='ignore')     # ignore overflow errors in the computation of p (when y/x is big)
        prob_nonzero=1/(np.cosh(y/x/self.param))**2
        np.seterr(over='warn')
        prob=np.zeros(self.dimensions)
        prob[:state[0]] = prob_nonzero
        return prob
        
    #likelihood: compute the likelihood of a given observation y assuming that the target is est_target

    def likelihood(self, y, est_target):
        p=self.bernoulli_param(self.state, est_target)
        if self.state==est_target:     # this is because the source is a terminal state, so 
            return 0                   # the state just visited was not certainly the source regardless of y
        return p if y==1 else 1-p

    # likelihood of observing "obs" if I am in state "state", vectorial in the source position
    def likelihood_vectorial(self, obs, state):
        p = self.bernoulli_param_vectorial(state)
        if obs==1:
            lkh = p
        else:
            lkh = 1 - p
        lkh[state] = 0
        return lkh
    


    #move: returns the state reached after applying the action "action" from the current state

    def move(self, action):
        new_state=self.state[0]+action[0], self.state[1]+action[1]
        #if the new position is outside the grid the state is not updated
        if new_state[0]<self.dimensions[0] and new_state[0]>=0 and new_state[1]<self.dimensions[1] and new_state[1]>=0:
            return new_state
        else:
            return self.state

   
   
    #advance: apply action and transition to the new state

    def advance(self, action):
        self.state = self.move(action)
        self.done = (self.state==self.source)
        

    #observe: draw a real (random) observation from the environment

    def observe(self):
        p=self.bernoulli_param(self.state, self.source)
        return 1 if np.random.uniform()<p else 0
        
    #greedy: estimates the target position as the argmax of the belief
    
    def greedy(self):
        index=np.random.choice(np.flatnonzero(self.belief == self.belief.max()))  # in case of tie (the belief has multiple maxima) I choose randomly (otherwise "index=np.argmax(self.belief)" would pick always the first element and this introduces a bias)
        self.estimated_target = np.unravel_index(index, self.dimensions)
        #if np.count_nonzero(self.belief == self.belief.max())>1:
        #    print("tie", np.count_nonzero(self.belief == self.belief.max()))       
        
         

    #chase: pick action to get one step closer to the current estimate of the target

    def chase(self):   # oppure potrei chiamarlo chase()
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
            