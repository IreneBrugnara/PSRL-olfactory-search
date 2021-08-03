import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import cosh
from common import Gridworld
import time

class Hybrid(Gridworld):
    def __init__(self, nrows, ncols, source, init_state, render=True, pause=0, param=0.2):
        super().__init__(nrows, ncols, source, init_state, render, pause, param)
        self.estimated_target=(None,None)     # current estimate of target position

        if self.render:
          self.scat_target = self.ax.scatter(self.estimated_target[1], self.estimated_target[0], color='b', marker='x', label="estimated target", s=150)
          
          
    def infotaxis_step(self, possible_actions, t):
        # non ho vettorizzato il calcolo perchè ci sono solo 5 elementi da valutare (only steps of length 1)
        #possible_actions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]  # set of candidate actions (still, right, left, down, up)

        new_states = list(map(self.move, possible_actions))
        num_actions = len(possible_actions)
        new_belief = np.empty((num_actions, 2), dtype=object) # one row for each new_state I could reach and one column for each observation (0 or 1) I could make
        # new_belief[i][y] is the belief I will have if I go to state new_states[i] and observe y
        # oppure potrei fare un dictionary con le keys che sono i new_states
        new_entropy = np.empty_like(new_belief, dtype=float)
        delta_S = np.empty(num_actions, dtype=float)
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
        # non serve che faccio l'update() perchè basta riciclare il calcolo già fatto sui 5
        
        
        return actual_obs  # only for the plot

    def update_entropy(self):
        self.entropy = - np.sum(self.belief[self.belief>0] * np.log(self.belief[self.belief>0]))

    #show: display a graphical representation of the grid

    def update_plots(self, t, obs):
        super().update_plots(t, obs)
        self.scat_target.set_offsets([self.estimated_target[1], self.estimated_target[0]])
        

    #thompson: thompson sampling to choose a new estimate for the target position

    def thompson(self):
        index=np.random.choice(self.dimensions[0]*self.dimensions[1], p=self.belief.ravel())
        self.estimated_target = np.unravel_index(index, self.dimensions)
        
        
    def chase(self):
        var_r=self.estimated_target[0]-self.state[0]
        var_c=self.estimated_target[1]-self.state[1]
        action_r=np.sign(var_r) # direction of movement along horizontal axis
        action_c=np.sign(var_c) # direction of movement along vertical axis
        if action_r==0 or action_c==0: # stay still on at least one axis
            return [(action_r, action_c)]
        else:
            return [(action_r, 0), (0, action_c)]
            
            



#gridworld_search: simulates Thompson or greedy algorithm and returns the number of steps t taken until target is reached

def hybrid_search(grid, tau, greedy=False, maxiter=np.inf, wait_first_obs=True, hybrid=False, prob=0.5):
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
            if grid.render:
                grid.show(t, obs)
            actions=grid.chase()
            if len(actions)==1:
                grid.advance(actions[0])
                obs=grid.observe()
                grid.update_efficient(obs)
            else:    # if two actions are equivalent, do infotaxis step
                if hybrid:
                    grid.update_entropy()
                    obs=grid.infotaxis_step(actions, t)
                else:   # ordinary thompson step
                    action = actions[0] if np.random.uniform()>prob else actions[1]
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
    
    
    
nrows=201
ncols=151
myseed=int(time.time())
np.random.seed(myseed)
print(myseed)
init_state = 185,75
real_target = 60,115
grid=Hybrid(nrows, ncols, real_target, init_state, render=True, pause=0, param=0.2)
print(hybrid_search(grid, 15, greedy=True, wait_first_obs=True, hybrid=False, prob=1))


# 123 -> 251
# 456 -> 195
# 789 -> 241
