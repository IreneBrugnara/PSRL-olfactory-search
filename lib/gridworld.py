import numpy as np
from itertools import product
from math import cosh
from animation.render import Render

class Gridworld():   # Abstract base class
    def __init__(self, size=(201,151), source=(60,115), init_state=(195,75), param=0.2, plot=False, pause=0):
        self.dimensions=size    # dimensions (number of rows and columns) of the grid
        self.state=init_state    # current state
        self.belief=np.ones(size)/(size[0]*size[1])    # uniform prior
        self.param=param    # parameter v/U in the observation model
        self.done=(source==init_state)    # flag set to true when real target is reached (reward=1)
        self.source=source    # real target position
        self.plot=plot   # if true, a graphical representation of the gridworld is printed
        self.possible_actions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]  # set of candidate actions (still, right, left, down, up)

        if plot:  # display a real-time animation
            self.render = self.construct_render(pause)  # factory
            
            
    def construct_render(self, pause):
        return Render(pause, self.belief, self.source, self.init_state)

    def show(self, t, obs, action=None): # the action is actually used in some subclasses
        self.render.show(t, obs, self.belief, self.state)
    
    
    #field: display the model of observations
    
    def field(self):
        field=np.empty_like(self.belief)
        nrows, ncols=self.dimensions
        for pos in product(range(nrows), range(ncols)):    # loop over each state in the grid
            field[pos] = self.bernoulli_param(pos, self.source)
        self.render.field(field, self.state, self.source)
        
        
    #update_efficient: posterior calculation given an observation, that vectorizes the computation of the likelihood with Numpy
    
    def update_belief(self, observation):
        self.belief = self.belief * self.likelihood_vectorial(observation, self.state)
        self.belief/=np.sum(self.belief)    # normalize

        
    
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

    
    #bernoulli_param_vectorial: parameter of the Bernoulli, vectorial in the source position
    
    def bernoulli_param_vectorial(self, state):
        nrows, ncols=self.dimensions    # like belief.shape
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
        

    # likelihood_vectorial: compute the likelihood of observing "obs" if I am in state "state", vectorial in the source position
    
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
        
          
    # first_update: update for the initial belief, just put to zero the belief in the initial state
    
    def first_update(self):
        self.belief[self.state] = 0
        self.belief/=np.sum(self.belief)
                

    # start_search: do the first observation before the search starts
                
    def start_search(self, wait_first_obs):
        obs=0
        if wait_first_obs:  # stay still until first observation
            while obs==0:
                obs=self.observe()
            self.update_belief(obs)  
        else:
            self.first_update()
        return obs
        
        
    #loop: one iteration of the search algorithm
        
    def loop(self, t, obs):  # passing in the observation because I need to plot with the obs at the previous iteration
        action=self.policy()
        if self.plot:
            self.show(t, obs, action)
        self.advance(action)
        obs=self.observe()
        self.update_belief(obs)
        return obs
        
        
        
    #search: abstract method, simulates search algorithm and returns the number of steps t taken until target is reached
    
    def search(self, max_iter=np.inf, wait_first_obs=True):
        obs = self.start_search(wait_first_obs)
        t=0    # time
        while not self.done and t<max_iter:
            obs = self.loop(t, obs)  # executes one step of search
            t += 1
        if self.plot:
            self.show(t, obs)
        return t
        
        
