from gridworld import *
import pickle

nrows=40
ncols=30
dimensions=(nrows,ncols)

n_trials=100
max_iter=1000

tau=np.concatenate((np.array([1]), np.arange(5,101,5)))
n_tau=len(tau)
times_thompson=np.zeros((n_tau,n_trials)) #number of steps taken by Thompson algorithm
times_greedy=np.zeros((n_tau,n_trials)) #number of steps taken by greedy algorithm

init_state = 35,15
real_target = 10,10

seed = 1
np.random.seed(seed)

for j in range(n_trials):
    for k in range(n_tau):
        grid=Gridworld(nrows, ncols, real_target, init_state, render=False)
        times_thompson[k,j]=gridworld_search(grid, tau[k], greedy=False, maxiter=max_iter)
        grid=Gridworld(nrows, ncols, real_target, init_state, render=False)
        times_greedy[k,j]=gridworld_search(grid, tau[k], greedy=True, maxiter=max_iter)

# save data to pickle file
filename="data.pickle"
outfile = open(filename,'wb')
pickle.dump(dimensions, outfile)
pickle.dump(max_iter, outfile)
pickle.dump(init_state, outfile)
pickle.dump(real_target, outfile)
pickle.dump(seed, outfile)
pickle.dump(tau, outfile)
pickle.dump(times_thompson, outfile)
pickle.dump(times_greedy, outfile)
outfile.close()
