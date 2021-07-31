import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import cosh
from common import Gridworld
from time import time

class Infotaxis(Gridworld):
    def __init__(self, nrows, ncols, source, init_state, render=True, pause=0, param=0.2):
        super().__init__(nrows, ncols, source, init_state, render, pause, param)
        self.entropy=np.log(nrows*ncols)   # entropy of the current belief (initially a flat distribution)
        
        
    def infotaxis_step(self, t):
        # non ho vettorizzato il calcolo perchè ci sono solo 5 elementi da valutare (only steps of length 1)
        possible_actions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]  # set of candidate actions (still, right, left, down, up)
        new_states = list(map(self.move, possible_actions))
        new_belief = np.empty((5, 2), dtype=object) # one row for each new_state I could reach and one column for each observation (0 or 1) I could make
        # new_belief[i][y] is the belief I will have if I go to state new_states[i] and observe y
        # oppure potrei fare un dictionary con le keys che sono i new_states
        new_entropy = np.empty_like(new_belief, dtype=float)
        delta_S = np.empty(5, dtype=float)
        # predict/anticipate  how the entropy changes in the three possible future scenarios: the source is found
        # in new_state, or source is not found and an observation is made or not
        for i, new_state in enumerate(new_states):
            p = self.bernoulli_param_vectorial(new_state)
            # p[new_state] = 0 ? sì ma è automatico per come ho implementato la p
            mean_p = np.sum(self.belief * p)   # mean bernoulli parameter weighted with the current belief
            for y in [0,1]:  # loop over the possible outcomes
                lkh = p if y==1 else 1-p   # likelihood
                lkh[new_state] = 0
                posterior = self.belief * lkh    # posterior calculation with bayesian update
                posterior /= np.sum(posterior)   # marginalize

                new_belief[i,y] = posterior
                new_entropy[i,y] = - np.sum(posterior[posterior>0] * np.log(posterior[posterior>0]))
                # oppure usare scipy.stats.entropy applicato a posterior.flatten()
            # controllare che l'entropia sia sempre positiva!
            greedy_term = self.belief[new_state]*(-self.entropy)
            exploration_term = (1-self.belief[new_state])*(mean_p*(new_entropy[i,1]-self.entropy) +
                                                   (1-mean_p)*(new_entropy[i,0]-self.entropy))
            delta_S[i] = greedy_term + exploration_term    # formula (1) del paper
            # expected entropy variation
        # pick the move that maximizes the expected reduction in entropy, randomly breaking ties
        i = np.random.choice(np.flatnonzero(np.isclose(delta_S,delta_S.min(),atol=1e-14)))   # delta_S==min(delta_S) is not correct because of numerical floating-point errors
        self.state = new_states[i]
        #best_states = [s for s in new_states if abs(delta_S[s]-min(delta_S.values()))<1e-14]    
        
        self.done = (self.state==self.source)
        actual_obs=0   # solo per il plot
        if not self.done:
            actual_obs = self.observe()
            self.belief = new_belief[i][actual_obs]
            self.entropy = new_entropy[i][actual_obs]
            
        if self.render:
            grid.show(t, actual_obs)
            
        if i==0:
            print(self.entropy)
        # non serve che faccio l'update() perchè basta riciclare il calcolo già fatto sui 5
        
        
    def mls_step(self, t):   # Most Likely State (sarebbe il greedy con tau=1)
        self.greedy()
        action=self.chase()
        self.advance(action)
        obs=self.observe()
        self.update_efficient(obs)
        if self.render:
            self.show(t, obs)


#gridworld_search: simulates Infotaxis and returns the number of steps t taken until target is reached; with threshold>0 implements Dual Mode control (under entropy threshold, the greedy action is taken)

def infotaxis_search(grid, threshold=0, maxiter=np.inf, wait_first_obs=True):
    if wait_first_obs:
        obs=0
        while obs==0:
            obs=grid.observe()
        grid.update_efficient(obs)    # INDENT
    else:
        grid.first_update()
    t=0
    if grid.render:
         grid.show(t, 0)
    while not grid.done and t<=maxiter:
        t+=1    # andrebbe incrementato dopo lo step, ma è per il plot
        if grid.entropy>=threshold:
            grid.infotaxis_step(t)
        else:
            grid.mls_step(t)
        if t>=maxiter:
            return t
    return t
    
    
   
nrows=201
ncols=151
myseed=int(time())
np.random.seed(1627193450) 
print(myseed)
init_state = 185,75
source = 60,115
grid=Infotaxis(nrows, ncols, source, init_state, render=True, pause=0, param=0.3)
print(infotaxis_search(grid, threshold=0, wait_first_obs=True, maxiter=7000))

