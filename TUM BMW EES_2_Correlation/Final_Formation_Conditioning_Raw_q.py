import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

client = MongoClient('localhost:27017')
db = client.hiwi
collection = db.production_steps
data_temp = list(collection.find({"features": {"$exists": True}}))
data = pd.DataFrame(data_temp)

database = {}
database['features'] = {}

query_1 = {"$and": [{"type": "formation"}, {"substep": "soacking"}, {"features": {"$exists": True}}]}
query_1 = list(collection.find(query_1))
database['features']['soacking_state'] = []
database['features']['soacking_success'] = []
database['features']['soacking_start_voltage'] = []
for i in range(len(query_1)):
    database['features']['soacking_state'].append(query_1[i]['features'][0]['value'])
    database['features']['soacking_success'].append(query_1[i]['features'][1]['value'])
    database['features']['soacking_start_voltage'].append(query_1[i]['features'][2]['value'])

query_2 = {"$and": [{"type": "formation"}, {"substep": "precharge"}, {"features": {"$exists": True}}]}
query_2 = list(collection.find(query_2))
database['features']['precharge_start_voltage'] = []
database['features']['precharge_energy'] = []
database['features']['precharge_success'] = []
database['features']['precharge_continuity'] = []
for i in range(len(query_2)):
    database['features']['precharge_start_voltage'].append(query_2[i]['features'][0]['value'])
    database['features']['precharge_energy'].append(query_2[i]['features'][2]['value'])
    database['features']['precharge_success'].append(query_2[i]['features'][3]['value'])
    database['features']['precharge_continuity'].append(query_2[i]['features'][4]['value'])

query_3 = {"$and": [{"type": "formation"}, {"substep": "charge"}, {"features": {"$exists": True}}]}
query_3 = list(collection.find(query_3))
database['features']['charge_start_voltage'] = []
database['features']['charge_energy'] = []
database['features']['charge_success'] = []
database['features']['charge_continuity'] = []
for i in range(len(query_3)):
    database['features']['charge_start_voltage'].append(query_3[i]['features'][0]['value'])
    database['features']['charge_energy'].append(query_3[i]['features'][2]['value'])
    database['features']['charge_success'].append(query_3[i]['features'][3]['value'])
    database['features']['charge_continuity'].append(query_3[i]['features'][4]['value'])

query_4 = {"$and": [{"type": "formation"}, {"substep": "discharge"}, {"features": {"$exists": True}}]}
query_4 = list(collection.find(query_4))
database['features']['discharge_energy'] = []
database['features']['discharge_success'] = []
database['features']['discharge_continuity'] = []
for i in range(len(query_4)):
    database['features']['discharge_energy'].append(query_4[i]['features'][2]['value'])
    database['features']['discharge_success'].append(query_4[i]['features'][3]['value'])
    database['features']['discharge_continuity'].append(query_4[i]['features'][4]['value'])

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

query_5 = {"$and": [{"type": "crino_eol"}, {"substep": "qdot"}, {"features": {"$exists": True}}]}
query_5 = list(collection.find(query_5))
database['features']['qdot_after_24h'] = []
database['features']['qdot_continuity'] = []
database['features']['qdot_labels'] = []

for i in range(len(query_5)):
    database['features']['qdot_after_24h'].append(query_5[i]['features'][0]['value'])
    database['features']['qdot_continuity'].append(query_5[i]['features'][1]['value'])

labels = pd.read_csv("app/data/4_raw_q/labels.csv", delimiter=";")
labels = (labels.iloc[:, 1]).tolist()
database['features']['qdot_labels'] = labels

data = pd.DataFrame(database['features'])

data = data.drop(['soacking_success',  'precharge_success', 'charge_success', 'discharge_success'], axis=1)

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

