from thompson import *
from infotaxis import *
import pickle
from multiprocessing import Pool, cpu_count
from itertools import starmap 
from functools import partial


nrows=201
ncols=151
dimensions=(nrows,ncols)
n_trials=1000
max_iter=1000
init_state = 185,75
real_target = 60,115
wait=True
entr_threshold=1.5
obs_param=0.2

tau=np.concatenate((np.array([1]), np.arange(5,71,5)))  # list(range()) e poi append
n_tau=len(tau)

nproc=32



# do the simulations
if nproc==1: # serial version
    times_thompson=np.zeros((n_tau,n_trials)) #number of steps taken by Thompson algorithm
    times_greedy=np.zeros((n_tau,n_trials)) #number of steps taken by greedy algorithm
    times_infotaxis=np.zeros((n_tau,n_trials))
    
    for j in range(n_trials):
        for k in range(n_tau):
            grid=Thompson(nrows, ncols, real_target, init_state, render=False, param=obs_param)
            times_thompson[k,j]=thompson_search(grid, tau[k], greedy=False, maxiter=max_iter, wait_first_obs=wait)
            grid=Thompson(nrows, ncols, real_target, init_state, render=False, param=obs_param)
            times_greedy[k,j]=thompson_search(grid, tau[k], greedy=True, maxiter=max_iter, wait_first_obs=wait)
        times_infotaxis[k,j]=infotaxis_search(grid, maxiter=max_iter, wait_first_obs=wait)
        times_dualmode[k,j]=infotaxis_search(grid, threshold=threshold, maxiter=max_iter, wait_first_obs=wait)

else: # parallel version
    def f(t, j, greedy):  # run one simulation of thompson/greedy with tau=t
        np.random.seed()
        grid=Thompson(nrows, ncols, real_target, init_state, render=False, param=obs_param)
        time=thompson_search(grid, t, greedy=greedy, maxiter=max_iter, wait_first_obs=wait)
        return time
    def f2(j, threshold):  # run one simulation of infotaxis
        np.random.seed()
        grid=Infotaxis(nrows, ncols, real_target, init_state, render=False, param=obs_param)
        time=infotaxis_search(grid, threshold=threshold, maxiter=max_iter, wait_first_obs=wait)
        return time
    if __name__ == '__main__':
        domain=list(product(tau, range(n_trials)))
        domain2=range(n_trials)
        f_thompson = partial(f, greedy=False)  #lambda t,j: f(t,j,greedy=False)
        f_greedy = partial(f, greedy=True) #lambda t,j: f(t,j,greedy=True)
        f2_infotaxis = partial(f2, threshold=0)  #lambda t,j: f(t,j,greedy=False)
        f2_dualmode = partial(f2, threshold=entr_threshold)
        with Pool(nproc) as p:
            res_thompson=p.starmap(f_thompson, domain)    #, chunksize=1)
            res_greedy=p.starmap(f_greedy, domain)     #, chunksize=1)
            res_infotaxis=p.map(f2_infotaxis, domain2)
            res_dualmode=p.map(f2_dualmode, domain2)
        times_thompson=np.array(res_thompson).reshape((n_tau, n_trials))
        times_greedy=np.array(res_greedy).reshape((n_tau, n_trials))
        times_infotaxis=np.array(res_infotaxis)
        times_dualmode=np.array(res_dualmode)
    

    # IN REALTÀ NON SERVIREBBE NEANCHE USARE STARMAP, tanto la j non la uso



# save data to pickle file

filename="data.pickle"
outfile = open(filename,'wb')
pickle.dump(dimensions, outfile)
pickle.dump(max_iter, outfile)
pickle.dump(init_state, outfile)
pickle.dump(real_target, outfile)
pickle.dump(wait, outfile)
pickle.dump(entr_threshold, outfile)
pickle.dump(obs_param, outfile)
pickle.dump(tau, outfile)
pickle.dump(times_thompson, outfile)
pickle.dump(times_greedy, outfile)
pickle.dump(times_infotaxis, outfile)
pickle.dump(times_dualmode, outfile)
outfile.close()



