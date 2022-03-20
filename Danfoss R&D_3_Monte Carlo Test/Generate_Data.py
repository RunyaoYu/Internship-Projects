"""
@author: Runyao Yu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import os

"""
---------------------------------------------------------------------------------------------------------------------------
                     ↓↓↓ Parameters setting ↓↓↓ 
"""
P1_mean,P1_std = 2.2, 0.1 # set mean and std for 1st feature: Th?
P2_mean,P2_std = 0, 1     # set mean and std for 2nd feature: Css?
P3_mean,P3_std = 0, 1     # set mean and std for 3rd feature: ?
path = "/Users/rampageyao/Desktop/丹佛斯工作" # set the path where you want to save the csv file later
dev_std = 6         # set upperbound and lowerbound for each feature: random values generated beyond 6 std will be deleted
N = 1000            # set how many points you want to generate
num_chip = 8        # set how many chips exsit in one module
bin = 1000          # no need to change: visualization variable
"""
                     ↑↑↑ Parameters setting ↑↑↑
---------------------------------------------------------------------------------------------------------------------------
"""

def generate_stochastic(P1_mean, P1_std, P2_mean, P2_std, P3_mean, P3_std, dev_std, N): 
    P1_lb, P1_ub = P1_mean - np.multiply(dev_std, P1_std), P1_mean + np.multiply(dev_std, P1_std)
    P2_lb, P2_ub = P2_mean - np.multiply(dev_std, P2_std), P2_mean + np.multiply(dev_std, P2_std)
    P3_lb, P3_ub = P3_mean - np.multiply(dev_std, P3_std), P3_mean + np.multiply(dev_std, P3_std)

    P1 = np.random.normal(P1_mean,P1_std,N).tolist()
    P1_new = [i for i in P1 if i>=P1_lb and i<=P1_ub]
    P2 = np.random.normal(P2_mean,P2_std,N).tolist()
    P2_new = [j for j in P2 if j>=P2_lb and j<=P2_ub] 
    P3 = np.random.normal(P3_mean,P3_std,N).tolist()
    P3_new = [k for k in P3 if k>=P3_lb and k<=P3_ub] 
    return P1_new, P2_new, P3_new


def generate_group_chip(P1, P2, P3, num_chip, N):
    base_index = list(range(int(N/num_chip)))
    Features = {}
    for i in base_index:
        Features["Group{}".format(i)] = {}
        Features["Group{}".format(i)]["Feature 1"] = []
        Features["Group{}".format(i)]["Feature 2"] = []
        Features["Group{}".format(i)]["Feature 3"] = []
        for j in range(num_chip):
            Features["Group{}".format(i)]["Feature 1"].append(P1[i*num_chip+j])
            Features["Group{}".format(i)]["Feature 2"].append(P2[i*num_chip+j])
            Features["Group{}".format(i)]["Feature 3"].append(P3[i*num_chip+j])

    data = pd.DataFrame(data=Features)
    return data


def plot_distribution(P, mean, std, bin):
    # mycolor.space 可以查找好看的色号
    yhist, xhist, patches = plt.hist(P, bin, facecolor='#00C9A7', alpha=0.5) 
     # draw curve fits the hist
    y = norm.pdf(xhist, mean, std)
    plt.plot(xhist, y, '#EE8BA6')  
    plt.title(r'Histogram')
    plt.show()  


P1, P2, P3 = generate_stochastic(P1_mean, P1_std, P2_mean, P2_std, P3_mean, P3_std, dev_std, N)
data = generate_group_chip(P1, P2, P3, num_chip, N)

data.to_csv(os.path.join(path,r'Chip_different_Groups.csv')) # save csv from dataframe generated

print("\n")
print("check mean and lenght of feature 1:",np.mean(P1),len(P1)) # check
print("check mean and lenght of feature 2:",np.mean(P2),len(P2)) # check
print("check mean and lenght of feature 3:",np.mean(P3),len(P3)) # check
print("\n")
print(data.head()) # check
print("check the first element of Feature 1 from the last Group:",P1[992]) #check
plot_distribution(P1, P1_mean,P1_std, bin) # check
