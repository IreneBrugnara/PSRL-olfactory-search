import numpy as np
from gridworld import Gridworld
import time
from render_votes import RenderVotes

class ActionVoting(Gridworld):
    def __init__(self, nrows, ncols, source, init_state, plot=True, pause=0, param=0.2):
        super().__init__(nrows, ncols, source, init_state, plot, pause, param)
        self.votes = np.zeros(5)   # votes[i] is the probability that possible_actions[i] is optimal
        self.actions = ["still", "right", "left", "down", "up"]
        self.bars = RenderVotes(self.votes, self.actions)

    def actionvoting_step(self, p=0.5):
        possible_actions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]  # set of candidate actions (right, left, down, up)
        
        x = self.state[0]
        y = self.state[1]
        
        # for efficiency reasons, I group the states into regions of the state space where the optimal action is the same
        up_left = self.belief[:x,:y]
        up = self.belief[:x,y]
        up_right = self.belief[:x,y+1:]
        right = self.belief[x,y+1:]
        down_right = self.belief[x+1:,y+1:]
        down = self.belief[x+1:,y]
        down_left = self.belief[x+1:,:y]
        left = self.belief[x,:y]
        still = self.belief[x,y]
        
        # when two actions are equivalent in a certain state,
        # I assign vote p to the horizontal one and 1-p to the vertical one
        self.votes[0] = still    # action "still" will never be voted because the belief is always zero in the current state
        self.votes[1] = p*up_right.sum() + right.sum() + p*down_right.sum()
        self.votes[2] = p*up_left.sum() + left.sum() + p*down_left.sum()
        self.votes[3] = (1-p)*down_left.sum() + down.sum() + (1-p)*down_right.sum()
        self.votes[4] = (1-p)*up_left.sum() + up.sum() + (1-p)*up_right.sum()
        
        
        
        #print(np.flatnonzero(np.isclose(votes,votes.max(),atol=1e-14)))
        #print(np.argmax(votes))
        i = np.random.choice(np.flatnonzero(np.isclose(self.votes,self.votes.max(),atol=1e-12)))  # index of the most voted action
        #i=np.argmax(self.votes)
        # potrei metterlo in un metodo randomized_argmax() in common.py visto che lo uso più volte
        chosen_action = possible_actions[i]
        
        return chosen_action, i   # la "i" serve solo al render, dovrei piuttosto fare un mapping (un dizionario o un array) fra le azioni e gli indici oppure con delle stringhe....
        
    
    

def actionvoting_search(grid, maxiter=np.inf, wait_first_obs=True, prob=0.5):
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
        action,i = grid.actionvoting_step(prob)
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
np.random.seed(34455555)
print(myseed)
init_state = 185,75
real_target = 60,115
grid=ActionVoting(nrows, ncols, real_target, init_state, plot=True, pause=2, param=0.2)
print(actionvoting_search(grid, wait_first_obs=False, prob=0.7))

