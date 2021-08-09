import numpy as np
import matplotlib.pyplot as plt
from render import Render
from gridworld import Gridworld

class RenderWithEstTarget(Render):
    def __init__(self, pause, belief, source, state):
        super().__init__(pause, belief, source, state)
        self.scat_target = self.ax.scatter(None, None, color='b', marker='x', label="estimated target", s=150)
        self.ax.legend(loc=(-0.45,0.5))

    def update_plots(self, t, obs, belief, state, estimated_target):
        super().update_plots(t, obs, belief, state)
        self.scat_target.set_offsets([estimated_target[1], estimated_target[0]])
        
    def show(self, t, obs, belief, state, estimated_target):
        self.update_plots(t, obs, belief, state, estimated_target)
        self.redraw()    

class Thompson(Gridworld):
    def __init__(self, nrows, ncols, source, init_state, plot=True, pause=0, param=0.2):
        super().__init__(nrows, ncols, source, init_state, plot, pause, param)
        
    def show(self, t, obs):
        self.render.show(t, obs, self.belief, self.state, self.estimated_target)
        
    def construct_render(self, pause):
        return RenderWithEstTarget(pause, self.belief, self.source, self.state)
        
        

#gridworld_search: simulates Thompson or greedy algorithm and returns the number of steps t taken until target is reached

def thompson_search(grid, tau, greedy=False, maxiter=np.inf, wait_first_obs=True, hybrid=False, prob=0.5):
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
            if grid.plot:
                grid.show(t, obs)
            actions=grid.chase()
            if len(actions)==1:
                grid.advance(actions[0])
                obs=grid.observe()
                grid.update_efficient(obs)
            else:    # if two actions are equivalent, do infotaxis between them
                if hybrid:
                    grid.update_entropy()
                    obs=grid.infotaxis_step(actions)
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
    if grid.plot:
        grid.show(t, obs)
    return t
