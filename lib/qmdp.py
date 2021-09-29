import numpy as np
from lib.gridworld import Gridworld
import time
from animation.render_votes import RenderVotes


class Qmdp(Gridworld):
    def __init__(self, size=(201,151), source=(60,115), init_state=(195,75), param=0.2, plot=False, pause=0,
                 gamma=0.9):
        super().__init__(size, source, init_state, param, plot, pause)
        self.votes = np.zeros(5)   # votes[i] is the weight of self.possible_actions[i]
        self.gamma = gamma    # parameter
        
    def construct_render(self, pause):
        return RenderVotes(pause, self.belief, self.source, self.state, self.possible_actions)

    def policy(self):
        new_states = list(map(self.move, self.possible_actions))
        
        # compute the state-action values from being in self.state and taking one action among possible_actions,
        # averaged over all possible source positions
        
        nrows, ncols=self.dimensions
        x = np.tile(np.arange(nrows).reshape(nrows,1), (1,ncols))
        y = np.tile(np.arange(ncols), (nrows,1))
        
        # for this MDP, the action-state value function depends only on the state you land on (called new_state)
        for i, new_state in enumerate(new_states):
            delta_x = new_state[0] - x
            delta_y = new_state[1] - y
            distance = np.abs(delta_x)+np.abs(delta_y)    # manhattan distance between new_state and all the possible sources
            values = self.gamma**distance 
            #values[self.state] = 0 not needed because self.belief[self.state]=0
            
            self.votes[i] = np.sum(self.belief * values)
            
        
        possibilities = np.flatnonzero(np.isclose(self.votes,self.votes.max(),atol=1e-15)) 
        
        i = np.random.choice(possibilities)  # index of the most voted action
        chosen_action = self.possible_actions[i]
        
        return chosen_action
        
    def show(self, t, obs, action=None):
        self.render.show(t, obs, self.belief, self.state, self.votes, action)    



