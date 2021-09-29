from lib.gridworld import Gridworld
from lib.thompson import Thompson
from lib.action_voting import ActionVoting
from lib.hybrid import Hybrid
from lib.dualmode import DualMode
from lib.thompson_adaptive import ThompsonAdaptive
from lib.qmdp import Qmdp
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import starmap, product
from functools import partial
from os import mkdir


size=201,151
n_trials=3000
max_iter=2000
init_state = 185,75
real_target = 60,115
wait=True
entr_threshold=1.5
obs_param=0.2

taus=np.concatenate((np.array([1]), np.arange(5,61,5)))  # list(range()) e poi append
n_tau=len(taus)

nproc=2#32



# do the simulations
if nproc==1: # serial version
    times_greedy_prob05=np.zeros((n_tau,n_trials))
    times_greedy_prob07=np.zeros((n_tau,n_trials))
    times_greedy_prob1=np.zeros((n_tau,n_trials))
    times_greedy_hybrid=np.zeros((n_tau,n_trials)) 
    
    times_infotaxis=np.zeros(n_trials)
    times_actionvoting=np.zeros(n_trials)
    times_qmdp=np.zeros(n_trials)
    times_greedy_adaptive=np.zeros(n_trials)

    for j in range(n_trials):
        for k in range(n_tau):
            grid=Thompson(size, real_target, init_state, plot=False, param=obs_param, tau=taus[k], greedy=True, prob=1)
            times_greedy_prob1[k,j]=grid.search(max_iter=max_iter, wait_first_obs=wait)
            grid=Thompson(size, real_target, init_state, plot=False, param=obs_param, greedy=True, tau=taus[k], prob=0.7)
            times_greedy_prob07[k,j]=grid.search(max_iter=max_iter, wait_first_obs=wait)
            grid=Thompson(size, real_target, init_state, plot=False, param=obs_param, tau=taus[k], greedy=True, prob=0.5)
            times_greedy_prob05[k,j]=grid.search(max_iter=max_iter, wait_first_obs=wait)
            grid=Hybrid(size, real_target, init_state, plot=False, param=obs_param, tau=taus[k], greedy=True)
            times_greedy_hybrid[k,j]=grid.search(max_iter=max_iter, wait_first_obs=wait)
        
        grid=DualMode(size, real_target, init_state, plot=False, param=obs_param, threshold=entr_threshold)
        times_infotaxis[j]=grid.search(max_iter=max_iter, wait_first_obs=wait)
        grid=ActionVoting(size, real_target, init_state, plot=False, param=obs_param, prob=0.7)
        times_actionvoting[j]=grid.search(max_iter=max_iter, wait_first_obs=wait)
        grid=Qmdp(size, real_target, init_state, plot=False, param=obs_param, gamma=0.8)
        times_qmdp[j]=grid.search(max_iter=max_iter, wait_first_obs=wait)
        grid=ThompsonAdaptive(size, real_target, init_state, plot=False, param=obs_param, coef=1.5, greedy=True)
        times_greedy_adaptive[j]=grid.search(max_iter=max_iter, wait_first_obs=wait)

else: # parallel version
    def f0(t, j, p=0.5):  # run one simulation of greedy
        np.random.seed(j)
        grid=Thompson(size, real_target, init_state, plot=False, param=obs_param, tau=t, greedy=True, prob=p)
        time=grid.search(max_iter=max_iter, wait_first_obs=wait)
        return time
    def f1(t, j):
        np.random.seed(j)
        grid=Hybrid(size, real_target, init_state, plot=False, param=obs_param, tau=t, greedy=True)
        time=grid.search(max_iter=max_iter, wait_first_obs=wait)
        return time
    def f2(j):  # run one simulation of infotaxis
        np.random.seed(j)
        grid=DualMode(size, real_target, init_state, plot=False, param=obs_param, threshold=entr_threshold)
        time=grid.search(max_iter=max_iter, wait_first_obs=wait)
        return time
    def f3(j):  # run one simulation of infotaxis
        np.random.seed(j)
        grid=ActionVoting(size, real_target, init_state, plot=False, param=obs_param, prob=0.7)
        time=grid.search(max_iter=max_iter, wait_first_obs=wait)
        return time
    def f4(j):
        np.random.seed(j)
        grid=Qmdp(size, real_target, init_state, plot=False, param=obs_param, gamma=0.8)
        time=grid.search(max_iter=max_iter, wait_first_obs=wait)
        return time
    def f5(j):
        np.random.seed(j)
        grid=ThompsonAdaptive(size, real_target, init_state, plot=False, param=obs_param, coef=1.5, greedy=True)
        time=grid.search(max_iter=max_iter, wait_first_obs=wait)
        return time
    if __name__ == '__main__':
        domain=list(product(taus, range(n_trials)))
        domain2=range(n_trials)

        with Pool(nproc) as p:
            res_greedy_prob1=p.starmap(partial(f0, p=1), domain)
            res_greedy_prob07=p.starmap(partial(f0, p=0.7), domain)
            res_greedy_prob05=p.starmap(partial(f0, p=0.5), domain)
            res_greedy_hybrid=p.starmap(f1, domain)
            res_infotaxis=p.map(f2, domain2)
            res_actionvoting=p.map(f3, domain2)
            res_qmdp=p.map(f4, domain2)
            res_greedy_adaptive=p.map(f5, domain2)
        times_greedy_prob1=np.array(res_greedy_prob1).reshape((n_tau, n_trials))
        times_greedy_prob07=np.array(res_greedy_prob07).reshape((n_tau, n_trials))
        times_greedy_prob05=np.array(res_greedy_prob05).reshape((n_tau, n_trials))
        times_greedy_hybrid=np.array(res_greedy_hybrid).reshape((n_tau, n_trials))
        times_infotaxis=np.array(res_infotaxis)
        times_actionvoting=np.array(res_actionvoting)
        times_qmdp=np.array(res_qmdp)
        times_greedy_adaptive=np.array(res_greedy_adaptive)

    # IN REALTÀ NON SERVIREBBE NEANCHE USARE STARMAP, tanto la j non la uso


# save data to pickle file
mkdir("Results_2")
filename="Results_2/data_2.pickle"
outfile = open(filename,'wb')
pickle.dump(size, outfile)
pickle.dump(max_iter, outfile)
pickle.dump(init_state, outfile)
pickle.dump(real_target, outfile)
pickle.dump(wait, outfile)
pickle.dump(entr_threshold, outfile)
pickle.dump(obs_param, outfile)
pickle.dump(taus, outfile)
pickle.dump(times_greedy_prob1, outfile)
pickle.dump(times_greedy_prob07, outfile)
pickle.dump(times_greedy_prob05, outfile)
pickle.dump(times_greedy_hybrid, outfile)
pickle.dump(times_infotaxis, outfile)
pickle.dump(times_actionvoting, outfile)
pickle.dump(times_qmdp, outfile)
pickle.dump(times_greedy_adaptive, outfile)
outfile.close()



