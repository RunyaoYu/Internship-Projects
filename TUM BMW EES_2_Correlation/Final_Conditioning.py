import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns

client = MongoClient('localhost:27017')
db = client.hiwi
collection = db.production_steps
data_temp = list(collection.find({"features": {"$exists": True}}))
data = pd.DataFrame(data_temp)

database = {}
database['features'] = {}
database['features']['conditioning_0_step_1_mean_voltage'] = []
database['features']['conditioning_0_step_3_energy'] = []
database['features']['conditioning_0_step_3_r_dc_step'] = []
database['features']['conditioning_0_step_3_r_dc'] = []
database['features']['conditioning_0_step_5_start_voltage'] = []
database['features']['conditioning_1_step_3_r_dc_step'] = []

for i in range(len(data_temp)):
    if data_temp[i]["cell_id"] != "Q_206":
        if data_temp[i]['substep'] == "cycle_0_step_1":
            database['features']['conditioning_0_step_1_mean_voltage'].append((data_temp[i]['features'][0]['value']))

        if data_temp[i]['substep'] == "cycle_0_step_3":
            if data_temp[i]['features'][2]['type'] == "energy":
                database['features']['conditioning_0_step_3_energy'].append((data_temp[i]['features'][2]['value']))
            if data_temp[i]['features'][3]['type'] == "r_dc_step":
                database['features']['conditioning_0_step_3_r_dc_step'].append((data_temp[i]['features'][3]['value']))
            if data_temp[i]['features'][4]['type'] == "r_dc":
                database['features']['conditioning_0_step_3_r_dc'].append((data_temp[i]['features'][4]['value']))

        if data_temp[i]['substep'] == "cycle_0_step_5":
            if data_temp[i]['features'][0]['type'] == "start_voltage":
                database['features']['conditioning_0_step_5_start_voltage'].append((data_temp[i]['features'][0]['value']))

        if data_temp[i]['substep'] == "cycle_1_step_3":
            if data_temp[i]['features'][3]['type'] == "r_dc_step":
                database['features']['conditioning_1_step_3_r_dc_step'].append((data_temp[i]['features'][3]['value']))

data = pd.DataFrame(database['features'])

def correlation_matrix(dataframe, th=0.6):
    corr_mat2 = dataframe.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(corr_mat2, annot=True, linewidths=.5, ax=ax)
    plt.show()
    corr_pairs = corr_mat2.unstack()
    sorted_pairs = corr_pairs.sort_values(kind="quicksort")
    strong_pairs = sorted_pairs[abs(sorted_pairs) > th]
    print('\n')
    print('High Correlation:')
    print(strong_pairs)


correlation_matrix(data, th=0.6)
