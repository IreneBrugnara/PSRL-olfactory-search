import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

# load data from pickle file
path="Results/narrower_cone_source_shifted/"
infile = open(path+"data.pickle",'rb')
dimensions = pickle.load(infile)
max_iter = pickle.load(infile)
init_state = pickle.load(infile)
real_target = pickle.load(infile)
wait = pickle.load(infile)
entr_threshold = pickle.load(infile)
param = pickle.load(infile)
tau = pickle.load(infile)
times_thompson = pickle.load(infile)
times_greedy = pickle.load(infile)
times_infotaxis = pickle.load(infile)
times_dualmode = pickle.load(infile)
infile.close()

# process data
n_trials = times_thompson.shape[1]
init_distance = abs(init_state[0]-real_target[0]) + abs(init_state[1]-real_target[1])

# compute statistics

avg_times_thompson = np.average(times_thompson, axis=1)
avg_times_greedy = np.average(times_greedy, axis=1)
avg_time_infotaxis = np.average(times_infotaxis)
avg_time_dualmode = np.average(times_dualmode)

median_times_thompson = np.median(times_thompson, axis=1)
median_times_greedy = np.median(times_greedy, axis=1)
median_time_infotaxis = np.median(times_infotaxis)
median_time_dualmode = np.median(times_dualmode)

best_idx_tau_greedy = np.argmin(avg_times_greedy)
best_idx_tau_thompson = np.argmin(avg_times_thompson)


# metadata
if sys.argv[1]=="file":   # write output to files instead to stdout
    f = open(path+'metadata.txt', 'w')
    sys.stdout = f
    
print("grid dimension: ", dimensions)
print("init state: ", init_state)
print("real target: ", real_target)
print("minimum search time possible: ", init_distance)
print("number of runs: ", n_trials)
print("max iter: ", max_iter)
print("wait first obs? ", wait)
print("entropy threshold: ", entr_threshold)
print("parameter of observation model: ", param)

if sys.argv[1]=="file":
    f.close()

# plot average times
plt.figure(figsize=(16, 9))
plt.plot(tau, avg_times_thompson, label="thompson")
plt.plot(tau, avg_times_greedy, label="greedy")
plt.hlines(y=avg_time_dualmode, xmin=tau[0], xmax=tau[-1], label="infotaxis", color='y', linestyle='--')
plt.hlines(y=init_distance, xmin=tau[0], xmax=tau[-1], label="minimum possible", color='m', linestyle=':')
plt.ylabel("search time")
plt.xlabel("tau")
plt.xticks(tau)
plt.title("average search time")
plt.legend()
if sys.argv[1]=="file":
    plt.savefig(path+"avg.png")
else:
    plt.show()

# plot median times
plt.figure(figsize=(16, 9))
plt.plot(tau, median_times_thompson, label="thompson")
plt.plot(tau, median_times_greedy, label="greedy")
plt.hlines(y=median_time_dualmode, xmin=tau[0], xmax=tau[-1], label="infotaxis", color='y', linestyle='--')
plt.hlines(y=init_distance, xmin=tau[0], xmax=tau[-1], label="minimum possible", color='m', linestyle=':')
plt.ylabel("search time")
plt.xlabel("tau")
plt.xticks(tau)
plt.title("median search time")
plt.legend(loc="upper right")
if sys.argv[1]=="file":
    plt.savefig(path+"med.png")
else:
    plt.show()


# plot histogram of search times
#bins=np.arange(0, max(np.max(times_greedy[0]), np.max(times_greedy[-1])), 2)
plt.figure(figsize=(16, 9))
begin=init_distance-50
cutoff=begin+300
bins=np.arange(init_distance-50, cutoff, 2)
plt.hist(times_greedy[0], bins=bins, alpha=0.5, label='tau='+str(tau[0]), density=False)
plt.hist(times_greedy[best_idx_tau_greedy], bins=bins, alpha=0.5, label='tau='+str(tau[best_idx_tau_greedy]), density=False)
plt.hist(times_greedy[-1], bins=bins, alpha=0.5, label='tau='+str(tau[-1]), density=False)
plt.title("pdf of search time, greedy")
plt.legend()
outliers_zero=np.count_nonzero(times_greedy[0]>cutoff)
outliers_best=np.count_nonzero(times_greedy[best_idx_tau_greedy]>cutoff)
plt.gcf().text(0.5,0.04,"outliers for tau="+str(tau[0])+": "+str(outliers_zero), ha='center')
plt.gcf().text(0.5,0.01,"outliers for tau="+str(tau[best_idx_tau_greedy])+": "+str(outliers_best), ha='center')
#print("how many times >= max_iter:", np.sum(times_greedy[0]>=max_iter))
if sys.argv[1]=="file":
    plt.savefig(path+"taus.png")
else:
    plt.show()


plt.figure(figsize=(16, 9))
cutoff=begin+300
bins=np.arange(begin, cutoff, 2)
plt.hist(times_greedy[best_idx_tau_greedy], bins=bins, alpha=0.5, label='tau='+str(tau[best_idx_tau_greedy])+" greedy", density=False)
plt.hist(times_thompson[best_idx_tau_thompson], bins=bins, alpha=0.5, label='tau='+str(tau[best_idx_tau_thompson])+" thompson", density=False)
plt.title("pdf of search time, greedy vs thompson with optimal tau")
plt.legend()
outliers_greedy=np.count_nonzero(times_greedy[best_idx_tau_greedy]>cutoff)
outliers_thompson=np.count_nonzero(times_thompson[best_idx_tau_thompson]>cutoff)
plt.gcf().text(0.5,0.04,"outliers for greedy: "+str(outliers_greedy), ha='center')
plt.gcf().text(0.5,0.01,"outliers for thompson: "+str(outliers_thompson), ha='center')
if sys.argv[1]=="file":
    plt.savefig(path+"pdf.png")
else:
    plt.show()


    
########################
plt.figure(figsize=(16, 9))
cutoff=1000
bins=np.arange(init_distance-50, cutoff+2, 2)
plt.hist(times_infotaxis, bins=bins, alpha=0.5, label='infotaxis without Dual Mode', density=False)
plt.hist(times_dualmode, bins=bins, alpha=0.5, label='infotaxis with Dual Mode', density=False)
plt.title("pdf of search time, Dual Mode")
plt.gcf().text(0.5,0.01,"entropy threshold = "+str(entr_threshold), ha='center')
plt.legend()
if sys.argv[1]=="file":
    plt.savefig(path+"pdf2.png")
else:
    plt.show()


plt.figure(figsize=(16, 9))
cutoff=1000
bins=np.arange(init_distance-50, cutoff+2, 2)
plt.hist(times_greedy[best_idx_tau_greedy], bins=bins, alpha=0.5, label='greedy', density=False)
plt.hist(times_dualmode, bins=bins, alpha=0.5, label='infotaxis', density=False)
plt.title("pdf of search time, infotaxis vs greedy with best tau")
plt.legend()
if sys.argv[1]=="file":
    plt.savefig(path+"pdf3.png")
else:
    plt.show()


