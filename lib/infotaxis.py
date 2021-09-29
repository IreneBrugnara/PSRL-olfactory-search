from lib.gridworld import Gridworld
import numpy as np
from animation.render_entropy import RenderEntropy
        

class Infotaxis(Gridworld):
    def __init__(self, size=(201,151), source=(60,115), init_state=(195,75), param=0.2, plot=False, pause=0):
        self.entropy=np.log(size[0]*size[1])   # entropy of the current belief (initially a flat distribution)
        super().__init__(size, source, init_state, param, plot, pause) 
        
        
    def construct_render(self, pause):
        return RenderEntropy(pause, self.belief, self.source, self.state, self.entropy)
        
    def show(self, t, obs):
        self.render.show(t, obs, self.belief, self.state, self.entropy)
        
        
    # infotaxis_step: chooses the one among the possible_actions that maximizes the expected reduction in entropy
    
    def infotaxis_step(self, possible_actions):
        new_states = list(map(self.move, possible_actions))   # only steps of length 1
        num_actions = len(possible_actions)
        new_belief = np.empty((num_actions, 2), dtype=object) # one row for each new_state I could reach and one column for each observation (0 or 1) I could make
        # new_belief[i][y] is the belief I will have if I go to state new_states[i] and observe y

        new_entropy = np.empty_like(new_belief, dtype=float)
        delta_S = np.empty(num_actions, dtype=float)
        # predict how the entropy changes in the three possible future scenarios: the source is found
        # in new_state, or source is not found and an observation is made or not
        for i, new_state in enumerate(new_states):
            p = self.bernoulli_param_vectorial(new_state)
            # p[new_state] = 0 is automatic given how p is implemented
            mean_p = np.sum(self.belief * p)   # mean bernoulli parameter weighted with the current belief
            for y in [0,1]:  # loop over the possible outcomes
                lkh = p if y==1 else 1-p   # likelihood
                lkh[new_state] = 0
                posterior = self.belief * lkh    # posterior calculation with bayesian update
                posterior /= np.sum(posterior)   # marginalize

                new_belief[i,y] = posterior
                new_entropy[i,y] = - np.sum(posterior[posterior>0] * np.log(posterior[posterior>0]))
                # it's like scipy.stats.entropy applied to posterior.flatten()
            greedy_term = self.belief[new_state]*(-self.entropy)
            exploration_term = (1-self.belief[new_state])*(mean_p*(new_entropy[i,1]-self.entropy) +
                                                   (1-mean_p)*(new_entropy[i,0]-self.entropy))
            delta_S[i] = greedy_term + exploration_term    # formula (1) of the paper
            # expected entropy variation
            
            
        # pick the move that maximizes the expected reduction in entropy, randomly breaking ties
        i = np.random.choice(np.flatnonzero(np.isclose(delta_S,delta_S.min(),atol=1e-14)))   # delta_S==min(delta_S) is not correct because of numerical floating-point errors
        self.state = new_states[i]
        
        self.done = (self.state==self.source)
        actual_obs=0   # only for the plot
        if not self.done:
            actual_obs = self.observe()
            self.belief = new_belief[i][actual_obs]    # no need to call update_belief(), I already have the new belief
            self.entropy = new_entropy[i][actual_obs]  # no need to call update_entropy(), I already have the new belief
        
        return actual_obs  # only for the plot


    
    def loop(self, t, obs):
        if self.plot:
            self.show(t, obs)
        obs=self.infotaxis_step(self.possible_actions)

            
            
    def update_entropy(self):
        self.entropy = - np.sum(self.belief[self.belief>0] * np.log(self.belief[self.belief>0]))
        
