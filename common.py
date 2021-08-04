import numpy as np
from itertools import product
from math import cosh
from render import Render

class Gridworld():
    def __init__(self, nrows, ncols, source, init_state, plot=True, pause=0, param=0.2):
        self.dimensions=(nrows,ncols)    # dimensions of the grid
        self.state=init_state    # current state
        self.belief=np.ones(self.dimensions)/(nrows*ncols)    # uniform prior
        self.param=param    # parameter v/U in the observation model
        self.done=(source==init_state)    # flag set to true when real target is reached (reward=1)
        self.source=source    # real target position
        self.plot=plot   # if true, a graphical representation of the gridworld is printed
        self.estimated_target=(None,None)     # current estimate of target position
        self.entropy=np.log(nrows*ncols)   # entropy of the current belief (initially a flat distribution)

        if plot:
            self.render = Render(pause, self.belief, source, init_state, self.estimated_target)

    def show(self, t, obs):
        self.render.show(t, obs, self.belief, self.state, self.estimated_target)
    
    #field: display the model of observations
        
    def field(self):
        field=np.empty_like(self.belief)
        nrows, ncols=self.dimensions
        for pos in product(range(nrows), range(ncols)):    # loop over each state in the grid
            field[pos] = self.bernoulli_param(pos, self.source)
        self.render.field(field, self.state, self.source)
        
        
    #update_efficient: posterior calculation given an observation, that vectorizes the computation of the likelihood with Numpy
    
    def update_efficient(self, observation):
        self.belief = self.belief * self.likelihood_vectorial(observation, self.state)
        self.belief/=np.sum(self.belief)

        
            
    
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

    
    def bernoulli_param_vectorial(self, state):
        nrows, ncols=self.dimensions    # oppure: belief.shape
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
        

    # likelihood_vectorial: compute the likelihood of a given observation y assuming that the target is est_target
    # likelihood of observing "obs" if I am in state "state", vectorial in the source position
    def likelihood_vectorial(self, obs, state):
        p = self.bernoulli_param_vectorial(state)
        if obs==1:
            lkh = p
        else:
            lkh = 1 - p
        lkh[state] = 0
        return lkh
    


    #move: returns the state reached after applying the action "action" from the current state

    def move(self, action):
        new_state=self.state[0]+action[0], self.state[1]+action[1]
        #if the new position is outside the grid the state is not updated
        if new_state[0]<self.dimensions[0] and new_state[0]>=0 and new_state[1]<self.dimensions[1] and new_state[1]>=0:
            return new_state
        else:
            return self.state

   
   
    #advance: apply action and transition to the new state

    def advance(self, action):
        self.state = self.move(action)
        self.done = (self.state==self.source)
        

    #observe: draw a real (random) observation from the environment

    def observe(self):
        p=self.bernoulli_param(self.state, self.source)
        return 1 if np.random.uniform()<p else 0
        
    #greedy: estimates the target position as the argmax of the belief
    
    def greedy(self):
        index=np.random.choice(np.flatnonzero(self.belief == self.belief.max()))  # in case of tie (the belief has multiple maxima) I choose randomly (otherwise "index=np.argmax(self.belief)" would pick always the first element and this introduces a bias)
        self.estimated_target = np.unravel_index(index, self.dimensions)
        #if np.count_nonzero(self.belief == self.belief.max())>1:
        #    print("tie", np.count_nonzero(self.belief == self.belief.max()))       
        
        
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
          
    # first_update: update for the initial belief, just put to zero the belief in the initial state
    
    def first_update(self):
        self.belief[self.state] = 0
        self.belief/=np.sum(self.belief)
        
    # infotaxis_step: chooses the one among the possible_actions that mamimizes the expected reduction in entropy
    
    def infotaxis_step(self, possible_actions):
        # non ho vettorizzato il calcolo perchè ci sono solo 5 elementi da valutare (only steps of length 1)
        
        new_states = list(map(self.move, possible_actions))
        num_actions = len(possible_actions)
        new_belief = np.empty((num_actions, 2), dtype=object) # one row for each new_state I could reach and one column for each observation (0 or 1) I could make
        # new_belief[i][y] is the belief I will have if I go to state new_states[i] and observe y
        # oppure potrei fare un dictionary con le keys che sono i new_states
        new_entropy = np.empty_like(new_belief, dtype=float)
        delta_S = np.empty(num_actions, dtype=float)
        # predict/anticipate  how the entropy changes in the three possible future scenarios: the source is found
        # in new_state, or source is not found and an observation is made or not
        for i, new_state in enumerate(new_states):
            p = self.bernoulli_param_vectorial(new_state)
            # p[new_state] = 0 ? sì ma è automatico per come ho implementato la p
            mean_p = np.sum(self.belief * p)   # mean bernoulli parameter weighted with the current belief
            for y in [0,1]:  # loop over the possible outcomes
                lkh = p if y==1 else 1-p   # likelihood
                lkh[new_state] = 0
                posterior = self.belief * lkh    # posterior calculation with bayesian update
                posterior /= np.sum(posterior)   # marginalize

                new_belief[i,y] = posterior
                new_entropy[i,y] = - np.sum(posterior[posterior>0] * np.log(posterior[posterior>0]))
                # oppure usare scipy.stats.entropy applicato a posterior.flatten()
            # controllare che l'entropia sia sempre positiva!
            greedy_term = self.belief[new_state]*(-self.entropy)
            exploration_term = (1-self.belief[new_state])*(mean_p*(new_entropy[i,1]-self.entropy) +
                                                   (1-mean_p)*(new_entropy[i,0]-self.entropy))
            delta_S[i] = greedy_term + exploration_term    # formula (1) del paper
            # expected entropy variation
        # pick the move that maximizes the expected reduction in entropy, randomly breaking ties
        i = np.random.choice(np.flatnonzero(np.isclose(delta_S,delta_S.min(),atol=1e-14)))   # delta_S==min(delta_S) is not correct because of numerical floating-point errors
        self.state = new_states[i]
        #best_states = [s for s in new_states if abs(delta_S[s]-min(delta_S.values()))<1e-14]    
        
        self.done = (self.state==self.source)
        actual_obs=0   # solo per il plot
        if not self.done:
            actual_obs = self.observe()
            self.belief = new_belief[i][actual_obs]
            self.entropy = new_entropy[i][actual_obs]            
        # non serve che faccio l'update() perchè basta riciclare il calcolo già fatto sui 5
        
        return actual_obs  # only for the plot
        
        
    def mls_step(self):   # Most Likely State (sarebbe il greedy con tau=1)
        self.greedy()
        action=self.chase()
        self.advance(action)
        obs=self.observe()
        self.update_efficient(obs)
        return obs  # only for the plot

    def update_entropy(self):
        self.entropy = - np.sum(self.belief[self.belief>0] * np.log(self.belief[self.belief>0]))
