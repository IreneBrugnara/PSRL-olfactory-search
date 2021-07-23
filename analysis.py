import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

# load data from pickle file
infile = open("data_targetsx.pickle",'rb')
dimensions = pickle.load(infile)
max_iter = pickle.load(infile)
init_state = pickle.load(infile)
real_target = pickle.load(infile)
wait = pickle.load(infile)
seed = pickle.load(infile)
tau = pickle.load(infile)
times_thompson = pickle.load(infile)
times_greedy = pickle.load(infile)
infile.close()

n_trials = times_thompson.shape[1]


# compute statistics

avg_times_thompson = np.average(times_thompson, axis=1)
avg_times_greedy = np.average(times_greedy, axis=1)

median_times_thompson = np.median(times_thompson, axis=1)
median_times_greedy = np.median(times_greedy, axis=1)

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
print("wait first obs?", wait)



# plot average times
plt.figure()
plt.plot(tau, avg_times_thompson, label="thompson")
plt.plot(tau, avg_times_greedy, label="greedy")
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


plt.figure()
cutoff=300
bins=np.arange(init_distance-50, cutoff, 2)
plt.hist(times_greedy[3], bins=bins, alpha=0.5, label='tau='+str(tau[3])+"greedy", density=False)
plt.hist(times_thompson[5], bins=bins, alpha=0.5, label='tau='+str(tau[5])+"thompson", density=False)
plt.title("pdf of search time, greedy vs thompson with optimal tau")
plt.legend()
print("outliers with greedy: ", np.count_nonzero(times_greedy[3]>cutoff))
print("outliers with thompson: ", np.count_nonzero(times_thompson[5]>cutoff))



if sys.argv[1]=="file":
    plt.savefig("pdf.png")
else:
    plt.show()

if sys.argv[1]!="file":
    print("outliers for tau="+str(tau[0])+": ", np.count_nonzero(times_greedy[0]>cutoff))
    print("outliers for tau="+str(tau[4])+": ", np.count_nonzero(times_greedy[9]>cutoff))

    print("how many times > max_iter:", np.sum(times_greedy[0]>max_iter))
