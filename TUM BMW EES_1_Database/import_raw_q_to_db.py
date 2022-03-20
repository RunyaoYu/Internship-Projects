import pandas as pd
import os
from pymongo import MongoClient
import numpy as np

client = MongoClient('localhost:27017')
db = client.hiwi

gain = [649878.14784727863, 652794.77764177881, 653061.224489796, 663349.91708126036, 652262.53567060747, 650671.00447336317]
offset = [-0.01069, -0.00928, -0.00648, 0.00146, 0.003425, -0.007]

path = os.path.join("app/data/4_raw_q/", "labels.csv")
csv_data = pd.read_csv(path, header=0, error_bad_lines=False, delimiter=';')
cell_names = np.array(csv_data.iloc[:, 0])
cell_labels = np.array(csv_data.iloc[:, 1])

label_data = []

for i in range(len(cell_names)):
    data = {}
    data["cell_name"] = "Q_"+cell_names[i][1:4]
    data["cell_label"] = int(cell_labels[i])
    label_data.append(data)


def extract_data(raw_data):
    diff = raw_data.iloc[:, 1]
    timeline = pd.to_datetime(raw_data.iloc[:, 0])
    return diff, timeline


for root, dirs, files in os.walk("app/data/4_raw_q/raw_q/"):
    for name in files:
        path = os.path.join(root, name)
        csv_data = pd.read_csv(path, header=1, error_bad_lines=False)

        data = {}
        index = name.rfind("_Q")
        data["cell_name"] = "Q_"+name[index+2:index+5]
        index = name.find("ch")
        data["channel"] = int(name[index+2:index+3])

        for items in label_data:
            if data["cell_name"] == items["cell_name"]:
                data["label"] = items["cell_label"]
                break

        diff, timeline = extract_data(csv_data)
        data["raw_q"] = {}
        data["raw_q"]["dpt_time"] = timeline.values.tolist()
        data["raw_q"]["step_time"] = np.linspace(0, len(data["raw_q"]["dpt_time"])*10, len(data["raw_q"]["dpt_time"])).tolist()

        data["raw_q"]["diff"] = np.multiply(np.add(np.array(diff), offset[data["channel"]-1]), gain[data["channel"]-1]).tolist()

        db.raw.insert_one(data)


db.client.close()
