import numpy as np
from animation.render_with_est_target import RenderWithEstTarget
from lib.gridworld import Gridworld



class Thompson(Gridworld):    # Most Likely State / Thompson with deep exploration
    def __init__(self, size=(201,151), source=(60,115), init_state=(195,75), param=0.2, plot=False, pause=0,
                 tau=1, greedy=True, prob=0.5):
        super().__init__(size, source, init_state, param, plot, pause)
        self.estimated_target=(None,None)     # current estimate of target position
        self.tau = tau  # exploration depth
        self.i = tau   # counter
        self.prob = prob  # probability of selecting horizontal action instead of vertical action in case of equivalent actions
        self.greedy = greedy  # when True, do MLS; when false, do Thompson
        
    def show(self, t, obs, action=None):
        self.render.show(t, obs, self.belief, self.state, self.estimated_target)
        
    def construct_render(self, pause):
        return RenderWithEstTarget(pause, self.belief, self.source, self.state)
        
    
    #mls: estimates the target position as the argmax of the belief (Maximum Likelihood State)
    
    def mls(self):
        index=np.random.choice(np.flatnonzero(self.belief == self.belief.max()))  # in case of tie (the belief has multiple maxima) I choose randomly (otherwise "index=np.argmax(self.belief)" would pick always the first element and this introduces a bias)
        self.estimated_target = np.unravel_index(index, self.dimensions)       
        
        
    #thompson: thompson sampling to choose a new estimate for the target position

    def thompson(self):
        index=np.random.choice(self.dimensions[0]*self.dimensions[1], p=self.belief.ravel())
        self.estimated_target = np.unravel_index(index, self.dimensions)
        
    
    #chase: pick action to get one step closer to the current estimate of the target

    def chase(self):
        var_r=self.estimated_target[0]-self.state[0]
        var_c=self.estimated_target[1]-self.state[1]
        action_r=np.sign(var_r) # direction of movement along horizontal axis
        action_c=np.sign(var_c) # direction of movement along vertical axis
        if action_r==0 or action_c==0: # stay still on at least one axis
            return [(action_r, action_c)]
        else:
            return [(action_r, 0), (0, action_c)]
             
        
    #policy: policy to choose next action based on current belief
        
    def policy(self):
        self.deep_explore()
        return self.break_ties()
        
        
    def deep_explore(self):
        if self.i==self.tau or self.state == self.estimated_target:   # resample every tau steps
            if self.greedy:
                self.mls()
            else:
                self.thompson()
            self.i = 1  # reset counter
        else:
            self.i += 1   # increment counter
            
    def break_ties(self):
        actions=self.chase()
        if len(actions)==1:
            action = actions[0]
        else:
            action = actions[0] if np.random.uniform()>self.prob else actions[1]  # solve ties
        return action
    

