import numpy as np
import matplotlib.pyplot as plt
from animation.render import Render
from lib.thompson import Thompson
   

class ThompsonAdaptive(Thompson):
    def __init__(self, size=(201,151), source=(60,115), init_state=(195,75), param=0.2, plot=False, pause=0,
                       greedy=True, prob=0.5, coef=1.5):
        super().__init__(size, source, init_state, param, plot, pause, tau=np.inf, greedy=greedy, prob=prob)
        self.current_max = 0
        self.coef = coef
        
        

    def deep_explore(self):
        if self.i==self.tau or self.state == self.estimated_target or self.belief.max() > self.current_max*self.coef:   # resample every tau steps
            if self.greedy:
                self.mls()
            else:
                self.thompson()
            self.current_max = self.belief.max()
            self.i = 1  # reset counter
        else:
            self.i += 1   # increment counter



