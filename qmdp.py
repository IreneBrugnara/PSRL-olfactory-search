import numpy as np
from gridworld import Gridworld
import time
from qmdp_votes import QmdpVotes

class Qmdp(Gridworld):
    def __init__(self, nrows, ncols, source, init_state, plot=True, pause=0, param=0.2):
        super().__init__(nrows, ncols, source, init_state, plot, pause, param)
        self.votes = np.zeros(5)   # votes[i] is the weight of possible_actions[i]
        self.actions = ["still", "right", "left", "down", "up"]
        self.bars = QmdpVotes(self.votes, self.actions)

    def qmdp_step(self, gamma=0.9):
        possible_actions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]
        new_states = list(map(self.move, possible_actions))
        
        # compute the state-action values from being in self.state and taking one action among possible_actions,
        # for all source positions
        
        # RICICLARE QUESTO CODICE DAL BERNOULLI_PARAM_VECTORIAL
        nrows, ncols=self.dimensions
        x = np.tile(np.arange(nrows).reshape(nrows,1), (1,ncols))
        y = np.tile(np.arange(ncols), (nrows,1))
        
        # for this MDP, the action-state value function depends only on the state you land on (called new_state)
        for i, new_state in enumerate(new_states):
            delta_x = new_state[0] - x
            delta_y = new_state[1] - y
            distance = np.abs(delta_x)+np.abs(delta_y)    # manhattan distance between new_state and all the possible sources
            values = gamma**distance    
            
            self.votes[i] = np.sum(self.belief * values)
            
        
        possibilities = np.flatnonzero(np.isclose(self.votes,self.votes.max(),atol=1e-12))
        if len(possibilities)>1:
            print(possibilities)
        i = np.random.choice(possibilities)  # index of the most voted action
        chosen_action = possible_actions[i]
        
        return chosen_action, i
        
        
        

def qmdp_search(grid, maxiter=np.inf, wait_first_obs=True, gamma=0.9):
    obs=0
    if wait_first_obs:
        while obs==0:
            obs=grid.observe()
        grid.update_efficient(obs)
    else:
        grid.first_update()
    t=0
    if grid.plot:
         grid.show(t, obs)
         #grid.bars.update(grid.votes)
    while not grid.done and t<=maxiter:
        t+=1
        action,i = grid.qmdp_step(gamma)
        grid.advance(action)
        obs = grid.observe()
        grid.update_efficient(obs)
        if grid.plot:
            grid.show(t, obs)
            grid.bars.update(grid.votes, i)
    return t


nrows=201
ncols=151
myseed=int(time.time())
np.random.seed(myseed)
print(myseed)
init_state = 185,75
real_target = 60,115
grid=Qmdp(nrows, ncols, real_target, init_state, plot=True, pause=0, param=0.2)
print(qmdp_search(grid, wait_first_obs=True, gamma=0.9))

