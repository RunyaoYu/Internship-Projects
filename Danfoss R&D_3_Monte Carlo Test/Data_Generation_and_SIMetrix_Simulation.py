"""
@author: Runyao Yu
runyao.yu@danfoss.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from pathlib import Path
from shutil import copy

"""
Data Generation
---------------------------------------------------------------------------------------------------------------------------
                     ↓↓↓ Parameters setting ↓↓↓ 
"""
P1_mean,P1_std = 2.2, 0.1   # set mean and std for 1st feature: Th?
P2_mean,P2_std = 3, 0.1     # set mean and std for 2nd feature: Css?
P3_mean,P3_std = 10, 0.1    # set mean and std for 3rd feature: ?
path = "C:/Users/U388316/OneDrive - Danfoss/Desktop/chip" # set the path where you want to save the csv file later
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
    data = data.transpose()
    return data


def plot_distribution(P, mean, std, bin):
    # mycolor.space 可以查找好看的色号
    yhist, xhist, patches = plt.hist(P, bin, facecolor='#00C9A7', alpha=0.5)  # alpha: how transparant
    # draw curve fits the hist
    y = norm.pdf(xhist, mean, std)
    plt.plot(xhist, y, '#EE8BA6')  
    plt.title(r'Histogram')
    plt.show()  


P1, P2, P3 = generate_stochastic(P1_mean, P1_std, P2_mean, P2_std, P3_mean, P3_std, dev_std, N)
data = generate_group_chip(P1, P2, P3, num_chip, N)

"""
Simulation Running
---------------------------------------------------------------------------------------------------------------------------
                     ↓↓↓ Parameters setting ↓↓↓ 
"""

path_fixed = "C:/Runyao/Fixed/" # stay the same
path_temp = "C:/Runyao/Temp/" # dynamically be changed
exe = r"C:/Program Files/SIMetrix840/bin64/SIMetrix.exe" # right click SIMetrix
sxscr = "C:/Runyao/Temp/HSDUT.sxsch.sxscr"

symbol_1 = "$$$parameter1$$$"
symbol_2 = "$$$parameter2$$$"
symbol_3 = "$$$parameter3$$$"

Vth = data["Feature 1"]
Css = data["Feature 2"] #tbc
Tth = data["Feature 3"] #tbc

path_store = "C:/Runyao/Store/HS" #文件夹会跟HS在同一个路径
from_path = r"C:/Runyao/Temp/" #从动态变化的文件夹copy到新的文件夹
inter_path = r"C:/Runyao/Store/HS_"#跟HS在同一个路径根据Group编号创建的文件夹
"""
                     ↑↑↑ Parameters setting ↑↑↑
---------------------------------------------------------------------------------------------------------------------------
"""


def run_simetrix(exe,sxscr): # run software: .exo + /s + refered
    cmd = '"' + exe + '" /s ' + sxscr
    stream = os.popen(cmd)
    output = stream.read()
    print(output)
    
    
def change_parameter(j, symbol_1, symbol_2, symbol_3, parameter_1, parameter_2, parameter_3, path_fixed, path_temp):
    with open(path_fixed+"TestingChip"+str(j)+".cir", 'r') as src:
        lines = src.readlines()
    lines = [x.replace(symbol_1, parameter_1) for x in lines]
    lines = [y.replace(symbol_2, parameter_2) for y in lines]    
    lines = [z.replace(symbol_3, parameter_3) for z in lines]    
    #with open("C:/Users/U388316/OneDrive - Danfoss/Desktop/chip/Group"+"{}".format(i)+"_"+"{}".format(j)+"th_Chip"+".lib", 'w') as dst:
    with open(path_temp+"TestingChip"+str(j)+".cir", 'w') as dst:
        dst.writelines(lines)


def create_folder(Vth, path_store):
    for i in range(len(Vth)):	#这里创建文件夹
    	# *定义一个变量判断文件是否存在,path指代路径,str(i)指代文件夹的名字*
        isExists = os.path.exists(path_store+"_"+str(i))
        if not isExists:						#判断如果文件不存在,则创建
            os.makedirs(path_store+"_"+str(i))	
        else:	
            continue			#如果文件不存在,则继续上述操作,直到循环结束
            
            
def copy_folder(from_path, to_path):
    #如果 to_path 目录不存在，则创建
    for file in from_path.iterdir():
        if file.is_file() and not (to_path / file.name).is_file():
            copy(file, to_path)


create_folder(Vth, path_store) #create 125 folders
for i in range(len(Vth)):#125 times
        """
        if read csv, lists in dataframe will become str, use following to deal with
        #temp1 = Vth[i].strip('][').split(', ') 
        #temp2 = Css[i].strip('][').split(', ')
        #temp3 = Tth[i].strip('][').split(', ') 
        """
        parameter_1 = Vth[i]
        parameter_2 = Css[i]
        parameter_3 = Tth[i]
        for j in range(len(parameter_1)): #8 times
            #update j_th chip of 8 chips in SiMetrix          
            change_parameter(j, symbol_1, symbol_2, symbol_3, str(parameter_1[j]), str(parameter_2[j]), str(parameter_3[j]), path_fixed, path_temp)
        """
        after this inner for loop, we created 8 different chips in i_th group
        """
        #run SiMetrix and output results for each group
        run_simetrix(exe,sxscr)
        to_path = inter_path+"{}".format(i)
        #print(to_path)
        copy_folder(Path(from_path), Path(to_path))
        