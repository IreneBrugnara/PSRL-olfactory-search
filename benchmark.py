from gridworld import *
import pickle

nrows=40
ncols=30

n_trials=1000

tau=np.arange(1,11)
n_tau=len(tau)
times_thompson=np.zeros((n_tau,n_trials)) #number of steps taken by Thompson algorithm
times_greedy=np.zeros((n_tau,n_trials)) #number of steps taken by greedy algorithm

init_state = 25,25
real_target = 10,10

np.random.seed(1)

for j in range(n_trials):
    for k in range(n_tau):
        grid=Gridworld(nrows, ncols, real_target, init_state, render=False)
        times_thompson[k,j]=gridworld_search(grid, tau[k])
        grid=Gridworld(nrows, ncols, real_target, init_state, render=False)
        times_greedy[k,j]=gridworld_search(grid, tau[k], greedy=True)

filename="data.pickle"
outfile = open(filename,'wb')
pickle.dump(times_thompson, outfile)
pickle.dump(times_greedy, outfile)
