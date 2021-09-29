import numpy as np
from lib.thompson import Thompson
from lib.infotaxis import Infotaxis

class Hybrid(Thompson, Infotaxis):

    def __init__(self, size=(201,151), source=(60,115), init_state=(195,75), param=0.2, plot=False, pause=0,
                 tau=1, greedy=True):
        super().__init__(size, source, init_state, param, plot, pause, tau, greedy)  # default prob=0.5


    def update_entropy(self):
        self.entropy = - np.sum(self.belief[self.belief>0] * np.log(self.belief[self.belief>0]))
        
        
    #loop: like thompson, but in case of tie between two actions, do infotaxis between them
    def loop(self, t, obs):
        self.deep_explore()
        actions=self.chase()
        if len(actions)==1:
            action = actions[0]
            self.advance(action)
            obs=self.observe()
            self.update_belief(obs)
            self.update_entropy()
        else:
            obs=self.infotaxis_step(actions)
            
        if self.plot:
            self.show(t, obs)
        return obs
