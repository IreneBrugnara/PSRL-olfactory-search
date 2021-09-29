import numpy as np
from lib.thompson import Thompson
from lib.infotaxis import Infotaxis
from animation.render_entropy import RenderEntropyThreshold

        

class DualMode(Infotaxis, Thompson):   # behaves like Infotaxis as long as the entropy is above a certain threshold, then switches to MLS policy

    
    def __init__(self, size=(201,151), source=(60,115), init_state=(195,75), param=0.2, plot=False, pause=0,
                 threshold=0):  # the default threshold=0 falls back to plain Infotaxis
        self.threshold = threshold
        super().__init__(size, source, init_state, param, plot, pause)    # default parameters tau=1, greedy=True, prob=0.5 for Thompson
    
    def construct_render(self, pause):
        return RenderEntropyThreshold(pause, self.belief, self.source, self.state, self.entropy, self.threshold)
        

    def loop(self, t, obs):
        if self.entropy>=self.threshold:
            obs=Infotaxis.loop(self, t, obs)
        else:
            obs=Thompson.loop(self, t, obs)
            self.update_entropy()   # only for the plot
        return obs



