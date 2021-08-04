import numpy as np
from itertools import product
from math import cosh
from render import Render

class Gridworld():
    def __init__(self, nrows, ncols, source, init_state, plot=True, pause=0, param=0.2):
        self.dimensions=(nrows,ncols)    # dimensions of the grid
        self.state=init_state    # current state
        self.belief=np.ones(self.dimensions)/(nrows*ncols)    # uniform prior
        self.param=param    # parameter v/U in the observation model
        self.done=(source==init_state)    # flag set to true when real target is reached (reward=1)
        self.source=source    # real target position
        self.plot=plot   # if true, a graphical representation of the gridworld is printed
        self.estimated_target=(None,None)     # current estimate of target position

        if plot:
            self.render = Render(pause, self.belief, source, init_state, self.estimated_target)

    def show(self, t, obs):
        self.render.show(t, obs, self.belief, self.state, self.estimated_target)
    
    #field: display the model of observations
        
    def field(self):
        field=np.empty_like(self.belief)
        nrows, ncols=self.dimensions
        for pos in product(range(nrows), range(ncols)):    # loop over each state in the grid
            field[pos] = self.bernoulli_param(pos, self.source)
        self.render.field(field, self.state, self.source)
        
        
    #update_efficient: posterior calculation given an observation, that vectorizes the computation of the likelihood with Numpy
    
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
        nrows, ncols=self.dimensions    # oppure: belief.shape
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
        

    # likelihood_vectorial: compute the likelihood of a given observation y assuming that the target is est_target
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

    def chase(self):
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
          
    # first_update: update for the initial belief, just put to zero the belief in the initial state
    
    def first_update(self):
        self.belief[self.state] = 0
        self.belief/=np.sum(self.belief)
