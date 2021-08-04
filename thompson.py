import numpy as np
import matplotlib.pyplot as plt



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
