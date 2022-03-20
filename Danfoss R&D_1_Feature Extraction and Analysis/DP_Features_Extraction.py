# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:23:21 2021

@author: Runyao Yu
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
import scipy
from scipy.integrate import simps
from scipy.signal import find_peaks

Vpk = [] 
Ipk = [] 
dV_dt = [] 
dI_dt = [] 
t_rise_V = [] 
t_fall_V = [] 
t_rise_I = [] 
t_fall_I = [] 
Eon = [] 
Eoff = [] 
temp_div_Vge = []
temp_div_Vce = []
temp_div_Ic = []

def extract_data(data):
    time = data.iloc[:,0]
    Vge = data.iloc[:,1]
    Vce = data.iloc[:,2]
    Ic = data.iloc[:,3]
    return time, Vge, Vce, Ic


def visualize(data):
    fig, axs = plt.subplots(ncols=1, nrows=3, constrained_layout=True, figsize=(5, 8), dpi = 1000)
    axs[0].set_title('1 - Vge')
    axs[1].set_title('2 - Vce')
    axs[2].set_title('3 - Ic')
    for i in range(len(axs)):
        axs[i].set_xlabel('Time in s')
        axs[i].set_ylabel('Values')
        axs[i].grid()
    for i in range(len(data)):
        axs[0].plot(data["Original"]["time"], data["Original"]["Vge"])
        axs[1].plot(data["Original"]["time"], data["Original"]["Vce"])
        axs[2].plot(data["Original"]["time"], data["Original"]["Ic"])
    plt.show()
    

def single_plot(y, x):
    plt.plot(x, y)
    plt.legend()
    plt.show()
    

def max_index(feature):
    max_value = max(feature)
    index = feature.index(max_value)
    return index


def gradient(y,x):
    dydx = np.gradient(y,x)
    return dydx


def integral(y,x):
    I = scipy.integrate.simps(y,x)
    return I


def extract_last (x):
     sec = x[(len(x)-1)]
     return sec


def proportional_time(time): # precision from datasheet is not enough, distribute it proportionally (time distance: 8ns in this case)
    transformed = np.linspace(time[0], extract_last(time), len(time))
    return transformed


def get_peak_voltage(voltage,time,t_init=0.00018):
    peaks, _ = find_peaks(voltage, height=500)
    vol_peaks = np.array(voltage)[peaks]
    time_peaks = (np.array(time)[peaks]).tolist()
    time_peak =  [np.array(time)[t] for t in peaks if np.array(time)[t] > t_init][0]
    time_index = time_peaks.index(time_peak)
    vol_peak = vol_peaks[time_index]
    return vol_peak, time_peak


def get_peak_current(current,time,t_init=0.00022):
    peaks, _ = find_peaks(current, height=1300)
    cur_peaks = np.array(current)[peaks]
    time_peaks = (np.array(time)[peaks]).tolist()
    time_peak =  [np.array(time)[t] for t in peaks if np.array(time)[t] > t_init][0]
    time_index = time_peaks.index(time_peak)
    cur_peak = cur_peaks[time_index]
    return cur_peak, time_peak


def get_stationary_current(current,time,time_Icpk,time_lenth=500e-9): # time_lenth may vary due to different types e.g. IGBT 500ns, SiC 100ns etc.
    for i in range(len(time)):
        if time[i] >=  time_Icpk + time_lenth:
            stationary_current = current[i]
            break
    return stationary_current


def get_stationary_voltage(voltage,time,time_Vcepk,time_lenth=500e-9): # time_lenth may vary due to different types e.g. IGBT 500ns, SiC 100ns etc.
    for i in range(len(time)):
        if time[i] >=  time_Vcepk + time_lenth:
            stationary_voltage = voltage[i]
            break
    return stationary_voltage


def get_t_rise_I(stationary_cur,current,time,percent1=0.1,percent2=0.9,t_init=0.00022):
    stationary_cur_v1 = percent1*stationary_cur
    stationary_cur_v2 = percent2*stationary_cur
    for i in range(len(current)):
        if current[i] >= stationary_cur_v1 and time[i] >= t_init:
            index1 = i
            break
    t1 = time[index1] 
    for j in range(len(current)):
        if current[j] >= stationary_cur_v2 and time[j] >= t_init:
            index2 = j
            break
    t2 = time[index2]
    t_rise_I = t2 - t1
    return t_rise_I,t1,t2,index1,index2


def get_t_rise_V(stationary_vol,voltage,time,percent1=0.1,percent2=0.9,t_init=0.00018):
    stationary_vol_v1 = percent1*stationary_vol
    stationary_vol_v2 = percent2*stationary_vol
    for i in range(len(voltage)):
        if voltage[i] >= stationary_vol_v1 and time[i] >= t_init:
            index1 = i
            break
    t1 = time[index1] 
    for j in range(len(voltage)):
        if voltage[j] >= stationary_vol_v2 and time[j] >= t_init:
            index2 = j
            break
    t2 = time[index2]
    t_rise_I = t2 - t1
    return t_rise_I,t1,t2,index1,index2


def derivative(F,t1,t2,index_t1,index_t2):
    dF_dt = (F[index_t2] - F[index_t1])/(t2 - t1)
    return dF_dt


def get_former_stationary_current(current,time,t1_V,time_lenth=500e-9):
    t_temp = t1_V - time_lenth
    for i in range(len(time)):
        if time[i] >= t_temp:
            former_stationary_cur = current[i]
            break
    return former_stationary_cur


def get_t_fall_I(former_stationary_cur,current,time,percent1=0.1,percent2=0.9,t_init=0.00018):
    former_stationary_cur_v1 = percent1*former_stationary_cur
    former_stationary_cur_v2 = percent2*former_stationary_cur
    for m in range(len(current)):
        if current[m] <= former_stationary_cur_v1 and time[m] >= t_init:
            index1 = m
            break
    t1 = time[index1]
    for n in range(len(current)):
        if current[n] <= former_stationary_cur_v2 and time[n] >= t_init:
            index2 = n
            break
    t2 = time[index2]
    t_fall_I = t1 - t2
    return t_fall_I,t1,t2,index1,index2
    

def get_former_stationary_voltage(voltage,time,t1_I,time_lenth=500e-9):
    t_temp = t1_I - time_lenth
    for i in range(len(time)):
        if time[i] >= t_temp:
            former_stationary_vol = voltage[i]
            break
    return former_stationary_vol


def get_t_fall_V(former_stationary_vol,voltage,time,percent1=0.1,percent2=0.9,t_init=0.00022):
    former_stationary_vol_v1 = percent1*former_stationary_vol
    former_stationary_vol_v2 = percent2*former_stationary_vol
    for m in range(len(voltage)):
        if voltage[m] <= former_stationary_vol_v1 and time[m] >= t_init:
            index1 = m
            break
    t1 = time[index1]
    for n in range(len(voltage)):
        if voltage[n] <= former_stationary_vol_v2 and time[n] >= t_init:
            index2 = n
            break
    t2 = time[index2]
    t_fall_V = t1 - t2
    return t_fall_V,t1,t2,index1,index2
    

def get_Eon(voltage,current,time,index_t1_I,f_index1_t1_V):
    temp_V = []
    temp_I = []
    temp_T = []
    for i in range(len(time)):
        if i >= index_t1_I and i <= f_index1_t1_V:
            temp_V.append(voltage[i])
            temp_I.append(current[i])
            temp_T.append(time[i])
    Eon = scipy.integrate.simps((np.multiply(np.array(temp_V),np.array(temp_I))).tolist(),temp_T)
    return Eon


def get_Eoff(voltage,current,time,index_t1_V,f_index1_t1_I):
    temp_V = []
    temp_I = []
    temp_T = []
    for i in range(len(time)):
        if i >= index_t1_V and i <= f_index1_t1_I:
            temp_V.append(voltage[i])
            temp_I.append(current[i])
            temp_T.append(time[i])
    Eoff = scipy.integrate.simps((np.multiply(np.array(temp_V),np.array(temp_I))).tolist(),temp_T)
    return Eoff


print('loading the csv files...')
for root, dirs, files in os.walk("C:/Users/U388316/OneDrive - Danfoss/Desktop/DanfossDP/DP"): # add the path, make sure that under the main folder there is a folder (specific requirement for os.walk)
    for name in files:
        path = os.path.join(root, name)
        csv_data = pd.read_csv(path, delimiter=',', header=0) # delimiter can be varied for different csv files; print column "time" to check the first element and choose header. time Length:60520 dtype:float64
        data = {}
        index = name.rfind("D")
        data["Module_name"] = name[index:index+5]
        data["Original"] = {}
        data["Feature"] = {}
        print('extracting the features for the csv file:', name[index:index+5],'...')
        
        time, Vge, Vce, Ic = extract_data(csv_data)
        data["Original"]["time"] = (proportional_time(time.values.round(64).tolist())).tolist()
        data["Original"]["Vge"] = Vge.values.round(64).tolist()
        data["Original"]["Vce"] = Vce.values.round(64).tolist()
        data["Original"]["Ic"] = Ic.values.round(64).tolist()
        
        Vol_Peak, time_Vol_Peak = get_peak_voltage(data["Original"]["Vce"],data["Original"]["time"],t_init=0.00018)
        Cur_Peak, time_Cur_Peak = get_peak_current(data["Original"]["Ic"],data["Original"]["time"],t_init=0.00022)
        Vpk.append(Vol_Peak)
        Ipk.append(Cur_Peak)
        data["Feature"]["Vce_peak"] = Vol_Peak
        data["Feature"]["Ic_peak"] = Cur_Peak
                
        stationary_cur = get_stationary_current(data["Original"]["Ic"],data["Original"]["time"],time_Cur_Peak,time_lenth=500e-9)
        t_rise_I_value, t1_I,t2_I, index_t1_I, index_t2_I = get_t_rise_I(stationary_cur,data["Original"]["Ic"],data["Original"]["time"],percent1=0.1,percent2=0.9,t_init=0.00022)
        t_rise_I.append(t_rise_I_value)
        data["Feature"]["t_rise_Ic"] = t_rise_I_value
        
        stationary_vol = get_stationary_voltage(data["Original"]["Vce"],data["Original"]["time"],time_Vol_Peak,time_lenth=500e-9)
        t_rise_V_value, t1_V, t2_V, index_t1_V, index_t2_V = get_t_rise_V(stationary_vol,data["Original"]["Vce"],data["Original"]["time"],percent1=0.1,percent2=0.9,t_init=0.00018)
        t_rise_V.append(t_rise_V_value)
        data["Feature"]["t_rise_Vce"] = t_rise_V_value
        
        former_stationary_cur = get_former_stationary_current(data["Original"]["Ic"],data["Original"]["time"],t1_V,time_lenth=500e-9)
        t_fall_I_value, f_t1_I,f_t2_I, f_index1_t1_I, f_index2_t2_I = get_t_fall_I(former_stationary_cur,data["Original"]["Ic"],data["Original"]["time"],percent1=0.1,percent2=0.9,t_init=0.00018)
        t_fall_I.append(t_fall_I_value)
        data["Feature"]["t_fall_Ic"] = t_fall_I_value
        
        former_stationary_vol = get_former_stationary_voltage(data["Original"]["Vce"],data["Original"]["time"],t1_I,time_lenth=500e-9)
        t_fall_V_value, f_t1_V,f_t2_V, f_index1_t1_V, f_index2_t2_V = get_t_fall_V(former_stationary_vol,data["Original"]["Vce"],data["Original"]["time"],percent1=0.1,percent2=0.9,t_init=0.00022)
        t_fall_V.append(t_fall_V_value)
        data["Feature"]["t_fall_Vce"] = t_fall_V_value
        
        dI_dt.append(derivative(data["Original"]["Ic"],t1_I,t2_I,index_t1_I,index_t2_I))
        dV_dt.append(derivative(data["Original"]["Vce"],t1_V,t2_V,index_t1_V,index_t2_V))
        data["Feature"]["dI_dt"] = derivative(data["Original"]["Ic"],t1_I,t2_I,index_t1_I,index_t2_I)
        data["Feature"]["dV_dt"] = derivative(data["Original"]["Vce"],t1_V,t2_V,index_t1_V,index_t2_V)
        
        Eon.append(get_Eon(data["Original"]["Vce"],data["Original"]["Ic"],data["Original"]["time"],index_t1_I,f_index1_t1_V))
        Eoff.append(get_Eoff(data["Original"]["Vce"],data["Original"]["Ic"],data["Original"]["time"],index_t1_V,f_index1_t1_I))
        data["Feature"]["Eon"] = get_Eon(data["Original"]["Vce"],data["Original"]["Ic"],data["Original"]["time"],index_t1_I,f_index1_t1_V)
        data["Feature"]["Eoff"] = get_Eoff(data["Original"]["Vce"],data["Original"]["Ic"],data["Original"]["time"],index_t1_V,f_index1_t1_I)
        
print('features have been succussfully extracted')
print('\n')
