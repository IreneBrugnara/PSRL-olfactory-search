import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import cosh
from scipy.stats import entropy as scipy_entropy


class Gridworld():
    def __init__(self, nrows, ncols, source, init_state, render=True, pause=0, param=0.2):
        self.dimensions=(nrows,ncols)    # dimensions of the grid
        self.state=init_state    # current state
        self.belief=np.ones(self.dimensions)/(nrows*ncols)    # uniform prior
        self.entropy=np.log(nrows*ncols)   # entropy of the current belief (initially a flat distribution)
        self.done=(source==init_state)    # flag set to true when real target is reached (reward=1)
        self.param=param    # parameter v/U in the observation model
        self.source=source    # real target position
        self.render=render    # if true, a graphical representation of the gridworld is printed
        self.pause=pause    # time interval between animation frames

        if self.render:
          self.fig = plt.figure(figsize=(16, 10))
          self.ax = self.fig.add_subplot(111)
          self.im = self.ax.imshow(self.belief, cmap='Greens')
          self.scat_source = self.ax.scatter(self.source[1], self.source[0], color='r', marker='*', label="real target", s=150)
          self.scat_me = self.ax.scatter(self.state[1], self.state[0], color='y', marker='o', s=5, zorder=0.1, label="visited positions")
          self.scat_mel = self.ax.scatter(self.state[1], self.state[0], color='orange', marker='o', label="current position", zorder=1, s=100)
          self.scat_obs = self.ax.scatter(None, None, color='purple', marker='o', label="observations", s=5, zorder=0.2)
          self.fig.colorbar(self.im, ax=self.ax, ticks=None)
          self.text = self.fig.text(0.4, 0.6, "")
          plt.show(block=False)

    #show: display a graphical representation of the grid

    def show(self, t, obs):
        self.im.autoscale()
        self.im.set_array(self.belief)
        self.scat_me.set_offsets(np.vstack([self.scat_me.get_offsets(), np.array([self.state[1],self.state[0]])]))
        self.scat_mel.set_offsets([self.state[1], self.state[0]])
        if obs:
            self.scat_obs.set_offsets(np.vstack([self.scat_obs.get_offsets(), np.array([self.state[1],self.state[0]])]))
            
        self.ax.set_title("t="+str(t))
        if t==0:
            self.ax.legend(bbox_to_anchor=(-0.2, 0.5))
        self.text.set_text("y=1" if obs else "")
        
        self.fig.canvas.draw()
        if self.pause!=0:     # pause=0 is the fastest animation
            plt.pause(self.pause)
        
    #field: display the model of observations
        
    def field(self):
        field=np.empty_like(self.belief)
        nrows, ncols=self.dimensions
        for pos in product(range(nrows), range(ncols)):    # loop over each state in the grid
            field[pos] = self.bernoulli_param(pos, self.source)
        plt.imshow(field)
        plt.colorbar()
        plt.show()
        
    
    
    #bernoulli_param: compute the parameter of the Bernoulli for the model of observations

    
    def bernoulli_param_vectorial(self, state):
        nrows, ncols=self.dimensions
        x = np.tile(np.arange(state[0]).reshape(state[0],1), (1,ncols))
        y = np.tile(np.arange(ncols), (state[0],1))
        x = state[0] - x
        y = state[1] - y
        np.seterr(over='ignore')     # ignore overflow errors in the computation of p (when y/x is big)
        prob_nonzero=1/(np.cosh(y/x/self.param))**2
        np.seterr(over='warn')
        prob=np.zeros(self.dimensions)
        prob[:state[0]] = prob_nonzero
        return prob
        

    #step: returns the state reached after applying the action "action" from the current state

    def step(self, action):
        new_state=self.state[0]+action[0], self.state[1]+action[1]
        #if the new position is outside the grid the state is not updated
        if new_state[0]<self.dimensions[0] and new_state[0]>=0 and new_state[1]<self.dimensions[1] and new_state[1]>=0:
            return new_state
        else:
            return self.state
        

        
        
    def infotaxis(self, t):  # chiamarla policy() oppure advance()
        # non ho vettorizzato il calcolo perchè ci sono solo 5 elementi da valutare
        possible_actions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]  # set of candidate actions (still, right, left, down, up)
        new_states = list(map(self.step, possible_actions))
        # a dictionary of lists: the keys of the dictionary are the possible new states and the indexes of the lists
        # are the possible observations
        new_belief = dict.fromkeys(new_states, [None,None])  # one row for each new_state I could reach and one column for each observation (0 or 1) I could make
        # new_belief[s][y] is the belief I will have if I go to state s and observe y
        # oppure potrei fare un dictionary con le keys che sono i new_states
        new_entropy = dict.fromkeys(new_states, [None,None])
        delta_S = dict.fromkeys(new_states)
        # predict/anticipate  how the entropy changes in the three possible future scenarios: the source is found
        # in new_state, or source is not found and an observation is made or not
        for s in new_states:
            p = self.bernoulli_param_vectorial(s)
            # p[new_state] = 0 ??????
            mean_p = np.sum(self.belief * p)   # mean bernoulli parameter weighted with the current belief

            
            for y in [0,1]:
                lkh = p if y==1 else 1-p   # likelihood
                lkh[s] = 0
                posterior = self.belief * lkh    # posterior calculation with bayesian update
                posterior /= np.sum(posterior)   # marginalize
                
                new_belief[s][y] = posterior
                new_entropy[s][y] = - np.nansum(posterior * np.log(posterior))
                # il nansum serve perchè dove il posterior è troppo piccolo il risultato della moltiplicazione è nan
                # oppure usare scipy.stats.entropy applicato a posterior.flatten()
            # controllare che l'entropia sia sempre positiva!
            greedy_term = self.belief[s]*(-self.entropy)
            exploration_term = (1-self.belief[s])*(mean_p*(new_entropy[s][1]-self.entropy) +
                                                   (1-mean_p)*(new_entropy[s][0]-self.entropy))
            # non mi piace che [s] viene usato in due modi diversi.... confonde le idee a leggerlo
            delta_S[s] = greedy_term + exploration_term    # formula (1) del paper
            # expected entropy variation
            
        # pick the move that maximizes the expected reduction in entropy
        self.state = min(delta_S, key=delta_S.get)
        self.done = (self.state==self.source)
        actual_obs=0   # solo per il plot
        if not self.done:
            actual_obs = self.observe()
            self.belief = new_belief[self.state][actual_obs]
            self.entropy = new_entropy[self.state][actual_obs]
            
        if self.render:
            grid.show(t, actual_obs)
        # ritorno una delle possible_actions
        
        # what to do in case of ties? randomizzo anche qua?
        
        
        # non serve che faccio l'update() perchè basta riciclare il calcolo già fatto sui 5
            

    #observe: draw a real (random) observation from the environment

    def observe(self):
        p=self.bernoulli_param(self.state, self.source)
        return 1 if np.random.uniform()<p else 0
        
        
    #bernoulli_param: compute the parameter of the Bernoulli for the model of observations
            
    def bernoulli_param(self, state, source):
        x=state[0]-source[0]
        y=state[1]-source[1]
        if x<=0:
            p=0   # no observations can be made behind the source
        else:
            if abs(y/x/self.param) < 350:    # to prevent overflow
                p=1/(cosh(y/x/self.param))**2
            else:
                p=0
        return p
        


#gridworld_search: simulates Infotaxis and returns the number of steps t taken until target is reached

def gridworld_search(grid, maxiter=np.inf, wait_first_obs=True):
    #if wait_first_obs:
    #    while obs==0:
    #        obs=grid.observe()
    #    grid.update_efficient(obs)    # INDENT
    t=0
    if grid.render:
         grid.show(t, 0)
    while not grid.done and t<=maxiter:
        print(t)
        grid.infotaxis(t)
        t+=1
    return t
    
    
nrows=200
ncols=200
myseed=311
np.random.seed(myseed)  
init_state = 130,100
source = 20,60
grid=Gridworld(nrows, ncols, source, init_state, render=True, pause=0)
print(gridworld_search(grid, wait_first_obs=True))

