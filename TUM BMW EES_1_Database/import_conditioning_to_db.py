import pandas as pd
import os
import numpy as np
from pymongo import MongoClient

client = MongoClient("localhost:27017")
db = client.hiwi

step_names = ["step_1", "step_2", "step_3", "step_4", "step_5", "step_6",
              "step_7", "step_8", "step_9", "step_10", "step_11"]


def get_cycle(data):
    line = data.iloc[:, 2].tolist()
    for i in range(len(line)-1):
        if line[i+1]-line[i] < 0:
            data.iloc[i+1:, 1] = data.iloc[i, 1] + 1
    num = set(data.iloc[:, 1])
    return len(num), data


def extract_data_from_step(step_data):
    line = step_data.iloc[:, 0].tolist()
    current = []
    voltage = []
    t_step = []
    dpt_time = []
    state = []

    for i in range(len(line)-1):
        current.append(step_data.iloc[i, 7])
        voltage.append(step_data.iloc[i, 8])
        t_step.append(step_data.iloc[i, 4])
        dpt_time.append(pd.to_datetime(step_data.iloc[i, 11]))
        state.append(step_data.iloc[i, 9])

        if line[i+1]-line[i] > 1:
            return current, voltage, t_step, dpt_time, state

    return current, voltage, t_step, dpt_time, state


for root, dirs, files in os.walk("app/data/2_cycles"):
    for name in files:
        print(name)
        path = os.path.join(root, name)
        csv_data = pd.read_csv(path, delimiter='\t', header=2)
        cycle, csv_data = get_cycle(csv_data)
        data = {}
        index = name.rfind("Q_")
        data["cell_name"] = name[index:index+5]
        data["conditioning"] = {}

        for ci in range(cycle):
            data["conditioning"][str(ci)] = {}
            temp_data = csv_data[csv_data.iloc[:, 1] == ci]
            for i in range(len(step_names)):
                current, voltage, t_step, dpt_time, state = extract_data_from_step(temp_data[temp_data.iloc[:, 2] == i+1])
                data["conditioning"][str(ci)][step_names[i]] = {}
                data["conditioning"][str(ci)][step_names[i]]["step_time"] = t_step
                data["conditioning"][str(ci)][step_names[i]]["dpt_time"] = dpt_time
                data["conditioning"][str(ci)][step_names[i]]["voltage"] = voltage

                for s in state:
                    if s == 'C' or s == 'R':
                        data["conditioning"][str(ci)][step_names[i]]["current"] = current
                    else:
                        data["conditioning"][str(ci)][step_names[i]]["current"] = np.multiply(np.array(current), -1.0).tolist()

        db.raw.insert_one(data)

db.client.close()
