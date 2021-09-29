import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
from math import cosh

part=2

# load data from pickle file
path="Results_2/"
infile = open(path+"data_2.pickle",'rb')
dimensions = pickle.load(infile)
max_iter = pickle.load(infile)
init_state = pickle.load(infile)
real_target = pickle.load(infile)
wait = pickle.load(infile)
entr_threshold = pickle.load(infile)
param = pickle.load(infile)
tau = pickle.load(infile)
times_greedy_prob1 = pickle.load(infile)
times_greedy_prob07 = pickle.load(infile)
times_greedy_prob05 = pickle.load(infile)
times_greedy_hybrid = pickle.load(infile)
times_infotaxis = pickle.load(infile)
times_actionvoting = pickle.load(infile)
times_qmdp = pickle.load(infile)
times_greedy_adaptive = pickle.load(infile)
infile.close()

# process data
n_trials = times_infotaxis.shape[0]
delta_y = init_state[1]-real_target[1]
delta_x = init_state[0]-real_target[0]
init_distance = abs(delta_y) + abs(delta_x)
init_prob = 1/(cosh(delta_y/delta_x/param))**2

# compute statistics

avg_times_greedy_prob1 = np.average(times_greedy_prob1, axis=1)
avg_times_greedy_prob07 = np.average(times_greedy_prob07, axis=1)
avg_times_greedy_prob05 = np.average(times_greedy_prob05, axis=1)
avg_times_greedy_hybrid = np.average(times_greedy_hybrid, axis=1)
avg_time_infotaxis = np.average(times_infotaxis)
avg_time_actionvoting = np.average(times_actionvoting)
avg_time_qmdp = np.average(times_qmdp)
avg_time_greedy_adaptive = np.average(times_greedy_adaptive)

median_times_greedy_prob1 = np.median(times_greedy_prob1, axis=1)
median_times_greedy_prob07 = np.median(times_greedy_prob07, axis=1)
median_times_greedy_prob05 = np.median(times_greedy_prob05, axis=1)
median_times_greedy_hybrid = np.median(times_greedy_hybrid, axis=1)
median_time_infotaxis = np.median(times_infotaxis)
median_time_actionvoting = np.median(times_actionvoting)
median_time_qmdp = np.median(times_qmdp)
median_time_greedy_adaptive = np.median(times_greedy_adaptive)
#best_idx_tau_greedy = np.argmin(avg_times_greedy)
#best_idx_tau_thompson = np.argmin(avg_times_thompson)



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
print("initial probability of detection: ", init_prob)

if sys.argv[1]=="file":
    f.close()

# plot average times
plt.figure(figsize=(16, 9))
plt.plot(tau, avg_times_greedy_prob1, label="greedy with p=1", color='orange')
plt.plot(tau, avg_times_greedy_prob07, label="greedy with p=0.7", color='limegreen')
plt.plot(tau, avg_times_greedy_prob05, label="greedy with p=0.5", color='c')
plt.plot(tau, avg_times_greedy_hybrid, label="greedy hybrid", color='r')
plt.hlines(y=avg_time_greedy_adaptive, xmin=tau[0], xmax=tau[-1], label="greedy adaptive", color='brown', linestyle='--')
#plt.hlines(y=avg_time_infotaxis, xmin=tau[0], xmax=tau[-1], label="infotaxis", color='y', linestyle='--')
#plt.hlines(y=avg_time_actionvoting, xmin=tau[0], xmax=tau[-1], label="action voting with p=0.7", color='m', linestyle='--')
#plt.hlines(y=avg_time_qmdp, xmin=tau[0], xmax=tau[-1], label="QMDP with gamma=0.8", color='b', linestyle='--')
#plt.hlines(y=init_distance, xmin=tau[0], xmax=tau[-1], label="minimum possible", color='k', linestyle=':')
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
plt.plot(tau, median_times_greedy_prob1, label="greedy with p=1", color='orange')
plt.plot(tau, median_times_greedy_prob07, label="greedy with p=0.7", color='limegreen')
plt.plot(tau, median_times_greedy_prob05, label="greedy with p=0.5", color='c')
plt.plot(tau, median_times_greedy_hybrid, label="greedy hybrid", color='r')
plt.hlines(y=median_time_greedy_adaptive, xmin=tau[0], xmax=tau[-1], label="greedy adaptive", color='brown', linestyle='--')
#plt.hlines(y=median_time_infotaxis, xmin=tau[0], xmax=tau[-1], label="infotaxis", color='y', linestyle='--')
#plt.hlines(y=median_time_actionvoting, xmin=tau[0], xmax=tau[-1], label="action voting with p=0.7", color='m', linestyle='--')
#plt.hlines(y=median_time_qmdp, xmin=tau[0], xmax=tau[-1], label="QMDP with gamma=0.8", color='b', linestyle='--')

#plt.hlines(y=init_distance, xmin=tau[0], xmax=tau[-1], label="minimum possible", color='k', linestyle=':')
plt.ylabel("search time")
plt.xlabel("tau")
plt.xticks(tau)
plt.title("median search time")
plt.legend()#loc="upper right")
if sys.argv[1]=="file":
    plt.savefig(path+"med.png")
else:
    plt.show()


# plot histogram of search times
#bins=np.arange(0, max(np.max(times_greedy[0]), np.max(times_greedy[-1])), 2)
plt.figure(figsize=(16, 9))
begin=init_distance-50
cutoff=begin+350
idx=3
best_tau=tau[idx]
bins=np.arange(init_distance-50, cutoff, 2)

outliers_1=np.count_nonzero(times_greedy_prob1[idx]>cutoff)
outliers_07=np.count_nonzero(times_greedy_prob07[idx]>cutoff)
outliers_05=np.count_nonzero(times_greedy_prob05[idx]>cutoff)
outliers_hyb=np.count_nonzero(times_greedy_hybrid[idx]>cutoff)
outliers_info=np.count_nonzero(times_infotaxis>cutoff)
outliers_av=np.count_nonzero(times_actionvoting>cutoff)
outliers_qmdp=np.count_nonzero(times_qmdp>cutoff)
outliers_greedy_adaptive=np.count_nonzero(times_greedy_adaptive>cutoff)


#plt.hist(times_greedy_prob1[idx], bins=bins, alpha=0.5, label="greedy with p=1 (outliers: "+str(outliers_1)+")", density=False, color='orange')
plt.hist(times_greedy_prob07[idx], bins=bins, alpha=0.5, label="greedy with p=0.7 (outliers: "+str(outliers_07)+")", density=False, color='limegreen')

#plt.hist(times_greedy_hybrid[idx], bins=bins, alpha=0.5, label="greedy hybrid (outliers: "+str(outliers_hyb)+")", density=False, color='r')

#plt.hist(times_actionvoting, bins=bins, alpha=0.5, label="action voting with p=0.7 (outliers: "+str(outliers_av)+")", density=False, color='m')
plt.hist(times_qmdp, bins=bins, alpha=0.5, label="QMDP with gamma=0.8 (outliers: "+str(outliers_qmdp)+")", density=False, color='b')
#plt.hist(times_greedy_adaptive, bins=bins, alpha=0.5, label="greedy adaptive (outliers: "+str(outliers_greedy_adaptive)+")", density=False, color='brown')


#plt.hist(times_greedy_prob05[idx], bins=bins, alpha=0.5, label="greedy with p=0.5 (outliers: "+str(outliers_05)+")", density=False, color='c')
plt.hist(times_infotaxis, bins=bins, alpha=0.5, label="infotaxis (outliers: "+str(outliers_info)+")", density=False, color='y')

plt.title("histogram of search time, tau="+str(best_tau))
plt.legend()
plt.xlabel("time")
plt.ylabel("counts")

if sys.argv[1]=="file":
    plt.savefig(path+"pdf.png", bbox_inches='tight', pad_inches=0.03)
else:
    plt.show()


