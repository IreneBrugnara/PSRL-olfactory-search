from gridworld import Gridworld
import numpy as np
from render_entropy import RenderEntropy

class Infotaxis(Gridworld):
    def __init__(self, nrows, ncols, source, init_state, plot=True, pause=0, param=0.2):
        super().__init__(nrows, ncols, source, init_state, plot, pause, param)
        
    def construct_render(self, pause):
        return RenderEntropy(pause, self.belief, self.source, self.state, self.entropy)
        
    def show(self, t, obs):
        self.render.show(t, obs, self.belief, self.state, self.entropy)
        
    def show_threshold(self, threshold):
        self.render.show_threshold(threshold)


#infotaxis_search: simulates Infotaxis and returns the number of steps t taken until target is reached; with threshold>0 implements Dual Mode control (under entropy threshold, the greedy action is taken)

def infotaxis_search(grid, threshold=0, maxiter=np.inf, wait_first_obs=True):
    obs=0
    if wait_first_obs:
        while obs==0:
            obs=grid.observe()
        grid.update_efficient(obs)    # INDENT
    else:
        grid.first_update()
    t=0
    if grid.plot:
        grid.show_threshold(threshold)
        grid.show(t, obs)
    possible_actions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]  # set of candidate actions (still, right, left, down, up)
    while not grid.done and t<=maxiter:
        t+=1    # andrebbe incrementato dopo lo step, ma è per il plot
        if grid.entropy>=threshold:
            obs=grid.infotaxis_step(possible_actions)
        else:
            obs=grid.mls_step()
            grid.update_entropy()   # only for the plot
        if grid.plot:
            grid.show(t, obs)
    return t
    
    
