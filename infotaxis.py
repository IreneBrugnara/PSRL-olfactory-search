import numpy as np
import matplotlib.pyplot as plt


def display_entropy(grid, entropy, threshold):
    grid.ax2 = plt.gcf().add_subplot(1,2,2)
    grid.entropy_level = grid.ax2.bar(0, entropy, width=0.1, color='pink', label='entropy')
    plt.ylim([0,entropy])
    #plt.hlines(y=threshold)#, xmin=tau[i][0], xmax=tau[i][-1], label="infotaxis", color='y', linestyle='--')
    grid.ax2.axhline(threshold, label='threshold')
    plt.legend()
    
def update_entropy_plot(grid, entropy):
    grid.entropy_level[0].set_height(entropy)


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
         grid.show(t, obs)
         display_entropy(grid, grid.entropy, threshold)
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
            update_entropy_plot(grid, grid.entropy)
    return t
    
    
