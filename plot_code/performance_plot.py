import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

##########################################################

num_iter = 30
gammas = 0.9

examples = ["tabular", "circular", "LQR", "nonlinear", "ARCH"]
path = "../data/"

algs = ["fvi.csv", "lstdboost.csv", "valueiter.csv"]



##########################################################


def plot(example, name, gamma):
    # loads the data from the data directory and plots the figure
    # example and name are strings to indicate what to load and what to name
    # see "examples" above for what the inputs are

    data = np.zeros((3, num_iter))

    for i in range(len(algs)):
        alg = algs[i]
        filename = str(gamma) + "_numiter" + str(num_iter) + "_"
        load = path + example + filename + alg
        data[i, :] = np.genfromtxt(load)

    data = data / data[0, 0]

    plt.plot(np.log(data[0, :]), "rp:", label = "Fitted Value Iteration")
    plt.plot(np.log(data[1, :]), "bo--", label= "Krylov-Bellman Boosting")
    plt.plot(np.log(data[2, :]), "gs-", label = "Value Iteration")
    plt.legend()
    plt.xlabel("Number of iterations", fontsize = 18)
    plt.ylabel("Log error", fontsize = 18)
    plt.title(name + " policy evaluation, $\gamma = $" + str(gamma), fontsize = 23)
    plt.savefig("plots/" + example + "_gamma" + str(gamma) + ".pdf", dpi = 300)
    plt.close()

##########################################################


names = ["Tabular", "Circular", "LQR", "Nonlinear", "ARCH"]

for i in range(len(names)):
    example = examples[i]
    name = names[i]

    plot(example, name, 0.9)
    plot(example, name, 0.99)

##########################################################

# plot with low samples

example = "nonlinear"
name = "Nonlinear"
gamma = 0.99

data = np.zeros((4, num_iter))

for i in range(len(algs)):
    alg = algs[i]
    filename = str(gamma) + "_numiter" + str(num_iter) + "_"
    load = path + example + filename + alg
    data[i, :] = np.genfromtxt(load)


alg = "lstdboost.csv"
filename = str(gamma) + "_numiter" + str(num_iter) + "_"
load = path + example + "_lowsamples_" + filename + alg
print(load)
data[3, :] = np.genfromtxt(load)



data = data / data[0, 0]

plt.plot(np.log(data[0, :]), "rp:", label = "Fitted Value Iteration")
plt.plot(np.log(data[1, :]), "bo--", label= "Krylov-Bellman Boosting")
plt.plot(np.log(data[2, :]), "gs-", label = "Value Iteration")
plt.plot(np.log(data[3, :]), "bx--", label = "Krylov-Bellman, low samples")
plt.legend(prop = {'size' : 8})
plt.xlabel("Number of iterations", fontsize = 18)
plt.ylabel("Log error", fontsize = 18)
plt.title(name + " policy evaluation, $\gamma = $" + str(gamma), fontsize = 23)
plt.savefig("plots/" + example + "wlow_gamma" + str(gamma) + ".pdf", dpi = 300)
plt.close()