import numpy as np
from lib.gridworld import Gridworld
import time
from animation.render_votes import RenderVotes


class ActionVoting(Gridworld):
    def __init__(self, size=(201,151), source=(60,115), init_state=(195,75), param=0.2, plot=False, pause=0,
                 prob=0.5):
        super().__init__(size, source, init_state, param, plot, pause)
        self.votes = np.zeros(5)   # votes[i] is the probability that self.possible_actions[i] is optimal
        self.prob = prob   # parameter

    def construct_render(self, pause):
        return RenderVotes(pause, self.belief, self.source, self.state, self.possible_actions)

    def policy(self):
        
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
        
        p = self.prob
        # when two actions are equivalent in a certain state,
        # I assign vote p to the horizontal one and 1-p to the vertical one
        self.votes[0] = still    # action "still" will never be voted because the belief is always zero in the current state
        self.votes[1] = p*up_right.sum() + right.sum() + p*down_right.sum()
        self.votes[2] = p*up_left.sum() + left.sum() + p*down_left.sum()
        self.votes[3] = (1-p)*down_left.sum() + down.sum() + (1-p)*down_right.sum()
        self.votes[4] = (1-p)*up_left.sum() + up.sum() + (1-p)*up_right.sum()
        

        i = np.random.choice(np.flatnonzero(np.isclose(self.votes,self.votes.max(),atol=1e-12)))  # index of the most voted action
        chosen_action = self.possible_actions[i]
        
        return chosen_action
        
    
    def show(self, t, obs, action=None):
        self.render.show(t, obs, self.belief, self.state, self.votes, action)
        
        

