import pickle
import matplotlib.pyplot as plt
import numpy as np


infile = open("data.pickle",'rb')
times_thompson = pickle.load(infile)
times_greedy = pickle.load(infile)

avg_times_thompson = np.average(times_thompson, axis=1)
avg_times_greedy = np.average(times_greedy, axis=1)

std_thompson = np.std(times_thompson, axis=1)
std_greedy = np.std(times_greedy, axis=1)



plt.figure()
tau=np.arange(1,11)
plt.errorbar(tau, avg_times_thompson, label="thompson")#, yerr=std_thompson)
plt.errorbar(tau, avg_times_greedy, label="greedy")#, yerr=std_greedy)
plt.ylabel("search time")
plt.xlabel("tau")
plt.xticks(tau)
plt.legend()
#plt.show()
plt.savefig("plot.png")


bins=np.arange(0, max(np.max(times_thompson[0]), np.max(times_thompson[4])), 2)
plt.hist(times_thompson[0], bins=bins, alpha=0.5, label='tau=1', density=False)
plt.hist(times_thompson[4], bins=bins, alpha=0.5, label='tau=5', density=False)
plt.title("pdf of search time, Thompson")
plt.legend()
#plt.show()
plt.savefig("pdf.png")
