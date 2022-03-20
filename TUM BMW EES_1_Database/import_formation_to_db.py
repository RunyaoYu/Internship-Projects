import pandas as pd
import os
from pymongo import MongoClient


client = MongoClient('localhost:27017')
db = client.hiwi

step_names = ["soacking", "precharge", "charge", "discharge"]


def extract_data_from_step(step_data):
    current = step_data.iloc[:, 7]
    voltage = step_data.iloc[:, 8]
    t_step = step_data.iloc[:, 4]
    dpt_time = pd.to_datetime(step_data.iloc[:, 11])
    state = step_data.iloc[:, 9]
    return current, voltage, t_step, dpt_time, state


for root, dirs, files in os.walk("app/data/1_formation"):

    for name in files:
        path = os.path.join(root, name)
        csv_data = pd.read_csv(path, delimiter='\t', header=1)

        data = {}
        index = name.rfind("Q_")
        data["cell_name"] = name[index:index+5]
        data["formation"] = {}

        for i in range(len(step_names)):
            current, voltage, t_step, dpt_time, state = extract_data_from_step(csv_data[csv_data.iloc[:, 2] == i+1])
            data["formation"][step_names[i]] = {}
            data["formation"][step_names[i]]["step_time"] = t_step.values.tolist()
            data["formation"][step_names[i]]["dpt_time"] = dpt_time.values.tolist()
            data["formation"][step_names[i]]["voltage"] = voltage.values.tolist()
            temp = state.values.tolist()
            for s in temp:
                if s == 'C' or s == 'R':
                    data["formation"][step_names[i]]["current"] = current.values.tolist()
                else:
                    data["formation"][step_names[i]]["current"] = (-current).values.tolist()

        db.raw.insert_one(data)

db.client.close()
