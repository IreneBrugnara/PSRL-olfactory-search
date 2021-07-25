import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

# load data from pickle file
infile = open("data.pickle",'rb')
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

n_trials = times_thompson.shape[1]


# compute statistics

avg_times_thompson = np.average(times_thompson, axis=1)
avg_times_greedy = np.average(times_greedy, axis=1)
avg_time_infotaxis = np.average(times_infotaxis)
avg_time_dualmode = np.average(times_dualmode)

median_times_thompson = np.median(times_thompson, axis=1)
median_times_greedy = np.median(times_greedy, axis=1)
median_time_infotaxis = np.median(times_infotaxis)
median_time_dualmode = np.median(times_dualmode)

print("avg_time_infotaxis", avg_time_infotaxis)
print("avg_time_dualmode", avg_time_dualmode)
print("median_time_infotaxis", median_time_infotaxis)
print("median_time_dualmode", median_time_dualmode)

if sys.argv[1]=="file":   # write output to files instead to stdout
    f = open('metadata.txt', 'w')
    sys.stdout = f
    
# metadata
print("grid dimension: ", dimensions)
print("init state: ", init_state)
print("real target: ", real_target)
init_distance = abs(init_state[0]-real_target[0]) + abs(init_state[1]-real_target[1])
print("minimum search time possible: ", init_distance)
print("number of runs: ", n_trials)
print("max iter: ", max_iter)
print("wait first obs? ", wait)
print("entropy threshold: ", entr_threshold)
print("parameter of observation model: ", param)

# plot average times
plt.figure()
plt.plot(tau, avg_times_thompson, label="thompson")
plt.plot(tau, avg_times_greedy, label="greedy")
plt.hlines(y=avg_time_dualmode, xmin=tau[0], xmax=tau[-1], label="dualmode", color='y', linestyle='--')
plt.ylabel("search time")
plt.xlabel("tau")
plt.xticks(tau)
plt.title("average search time")
plt.legend()
if sys.argv[1]=="file":
    plt.savefig("avg.png")
else:
    plt.show()

# plot median times
plt.figure()
plt.plot(tau, median_times_thompson, label="thompson")
plt.plot(tau, median_times_greedy, label="greedy")
plt.hlines(y=median_time_dualmode, xmin=tau[0], xmax=tau[-1], label="dualmode", color='y', linestyle='--')
plt.ylabel("search time")
plt.xlabel("tau")
plt.xticks(tau)
plt.title("median search time")
plt.legend()
if sys.argv[1]=="file":
    plt.savefig("med.png")
else:
    plt.show()


# plot histogram of search times
#bins=np.arange(0, max(np.max(times_greedy[0]), np.max(times_greedy[-1])), 2)
plt.figure()
cutoff=500
bins=np.arange(init_distance-50, cutoff, 2)
plt.hist(times_greedy[0], bins=bins, alpha=0.5, label='tau='+str(tau[0]), density=False)
plt.hist(times_greedy[4], bins=bins, alpha=0.5, label='tau='+str(tau[4]), density=False)
plt.hist(times_greedy[-1], bins=bins, alpha=0.5, label='tau='+str(tau[-1]), density=False)
plt.title("pdf of search time, greedy")
plt.legend()
if sys.argv[1]!="file":
    print("outliers for tau="+str(tau[0])+": ", np.count_nonzero(times_greedy[0]>cutoff))
    print("outliers for tau="+str(tau[4])+": ", np.count_nonzero(times_greedy[9]>cutoff))

    print("how many times >= max_iter:", np.sum(times_greedy[0]>=max_iter))
if sys.argv[1]=="file":
    plt.savefig("taus.png")
else:
    plt.show()

best_idx_tau_greedy = np.argmin(avg_times_greedy)
print("best tau for greedy: ", tau[best_idx_tau_greedy])
best_idx_tau_thompson = np.argmin(avg_times_thompson)
print("best tau for thompson: ", tau[best_idx_tau_thompson])

plt.figure()
cutoff=300
bins=np.arange(init_distance-50, cutoff, 2)
plt.hist(times_greedy[best_idx_tau_greedy], bins=bins, alpha=0.5, label='tau='+str(tau[best_idx_tau_greedy])+"greedy", density=False)
plt.hist(times_thompson[best_idx_tau_thompson], bins=bins, alpha=0.5, label='tau='+str(tau[best_idx_tau_thompson])+"thompson", density=False)
plt.title("pdf of search time, greedy vs thompson with optimal tau")
plt.legend()
print("outliers with greedy: ", np.count_nonzero(times_greedy[best_idx_tau_greedy]>cutoff))
print("outliers with thompson: ", np.count_nonzero(times_thompson[best_idx_tau_thompson]>cutoff))
if sys.argv[1]=="file":
    plt.savefig("pdf.png")
else:
    plt.show()


    
########################
plt.figure()
cutoff=1000
bins=np.arange(init_distance-50, cutoff+2, 2)
plt.hist(times_infotaxis, bins=bins, alpha=0.5, label='infotaxis', density=False)
plt.hist(times_dualmode, bins=bins, alpha=0.5, label='dualmode', density=False)
plt.title("pdf of search time, infotaxis vs dualmode")
plt.legend()
if sys.argv[1]=="file":
    plt.savefig("pdf2.png")
else:
    plt.show()


plt.figure()
cutoff=1000
bins=np.arange(init_distance-50, cutoff+2, 2)
plt.hist(times_greedy[best_idx_tau_greedy], bins=bins, alpha=0.5, label='greedy', density=False)
plt.hist(times_dualmode, bins=bins, alpha=0.5, label='dualmode', density=False)
plt.title("pdf of search time, dualmode vs greedy with best tau")
plt.legend()
if sys.argv[1]=="file":
    plt.savefig("pdf3.png")
else:
    plt.show()


