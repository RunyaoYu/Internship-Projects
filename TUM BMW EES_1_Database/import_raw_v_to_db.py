
import pandas as pd
import os
from pymongo import MongoClient
import numpy as np

client = MongoClient('localhost:27017')
db = client.hiwi


def extract_data(raw_data):
    Q_cell = pd.DataFrame(raw_data.iloc[:, 1:])

    for col in Q_cell.columns:
        beginning_substring = col[:1]
        ending_substring = col[1:]
        new_string = beginning_substring + '_' + ending_substring
        Q_cell.rename(columns={col: new_string}, inplace=True)

    timeline = pd.to_datetime(raw_data.iloc[:, 0])
    return Q_cell, timeline


for root, dirs, files in os.walk("app/data/3_raw_v/"):
    for name in files:
        path = os.path.join(root, name)
        csv_data = pd.read_csv(path)
        Q_cell, timeline = extract_data(csv_data)
        # cell_name = Q_cell column name
        for col in Q_cell.columns:
            data = {}
            data["cell_name"] = col
            data["raw_v"] = {}
            data["raw_v"]["dpt_time"] = timeline.values.tolist()
            # one minute interval
            data["raw_v"]["step_time"] = np.linspace(0, len(data["raw_v"]["dpt_time"])*60, len(data["raw_v"]["dpt_time"])).tolist()
            data["raw_v"]["voltage"] = Q_cell[col].values.tolist()
            db.raw.insert_one(data)

db.client.close()
