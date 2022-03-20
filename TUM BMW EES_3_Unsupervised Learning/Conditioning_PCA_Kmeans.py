import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

client = MongoClient('localhost:27017')
db = client.hiwi
collection = db.production_steps
data_temp = list(collection.find({"features": {"$exists": True}}))
data = pd.DataFrame(data_temp)

database = {}
database['conditioning_features'] = {}
index = []

for k in range(2):
    database['conditioning_features'][str(k)] = {}
    database['conditioning_features']['0']['conditioning_0_step_1_mean_voltage'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_3_start_voltage'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_3_duration'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_3_energy'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_3_r_dc_step'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_3_r_dc'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_4_start_voltage'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_4_duration'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_4_energy'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_5_start_voltage'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_5_duration'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_5_energy'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_6_start_voltage'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_6_duration'] = []
    database['conditioning_features'][str(k)]['conditioning_' + str(k) + '_step_6_energy'] = []

for i in range(len(data_temp)):
    if data_temp[i]["cell_id"] != "Q_206":
        if data_temp[i]['substep'] == "cycle_0_step_1":
            database['conditioning_features']['0']['conditioning_0_step_1_mean_voltage'].append((data_temp[i]['features'][0]['value']))
            index.append(data_temp[i]['cell_id'])
        if data_temp[i]['substep'] == "cycle_0_step_3":
            if data_temp[i]['features'][0]['type'] == "start_voltage":
                database['conditioning_features']['0']['conditioning_0_step_3_start_voltage'].append((data_temp[i]['features'][0]['value']))
            if data_temp[i]['features'][1]['type'] == "duration":
                database['conditioning_features']['0']['conditioning_0_step_3_duration'].append((data_temp[i]['features'][1]['value']))
            if data_temp[i]['features'][2]['type'] == "energy":
                database['conditioning_features']['0']['conditioning_0_step_3_energy'].append((data_temp[i]['features'][2]['value']))
            if data_temp[i]['features'][3]['type'] == "r_dc_step":
                database['conditioning_features']['0']['conditioning_0_step_3_r_dc_step'].append((data_temp[i]['features'][3]['value']))
            if data_temp[i]['features'][4]['type'] == "r_dc":
                database['conditioning_features']['0']['conditioning_0_step_3_r_dc'].append((data_temp[i]['features'][4]['value']))
        if data_temp[i]['substep'] == "cycle_0_step_4":
            if data_temp[i]['features'][0]['type'] == "start_voltage":
                database['conditioning_features']['0']['conditioning_0_step_4_start_voltage'].append((data_temp[i]['features'][0]['value']))
            if data_temp[i]['features'][1]['type'] == "duration":
                database['conditioning_features']['0']['conditioning_0_step_4_duration'].append((data_temp[i]['features'][1]['value']))
            if data_temp[i]['features'][2]['type'] == "energy":
                database['conditioning_features']['0']['conditioning_0_step_4_energy'].append((data_temp[i]['features'][2]['value']))
        if data_temp[i]['substep'] == "cycle_0_step_5":
            if data_temp[i]['features'][0]['type'] == "start_voltage":
                database['conditioning_features']['0']['conditioning_0_step_5_start_voltage'].append((data_temp[i]['features'][0]['value']))
            if data_temp[i]['features'][1]['type'] == "duration":
                database['conditioning_features']['0']['conditioning_0_step_5_duration'].append((data_temp[i]['features'][1]['value']))
            if data_temp[i]['features'][2]['type'] == "energy":
                database['conditioning_features']['0']['conditioning_0_step_5_energy'].append((data_temp[i]['features'][2]['value']))
        if data_temp[i]['substep'] == "cycle_0_step_6":
            if data_temp[i]['features'][0]['type'] == "start_voltage":
                database['conditioning_features']['0']['conditioning_0_step_6_start_voltage'].append((data_temp[i]['features'][0]['value']))
            if data_temp[i]['features'][1]['type'] == "duration":
                database['conditioning_features']['0']['conditioning_0_step_6_duration'].append((data_temp[i]['features'][1]['value']))
            if data_temp[i]['features'][2]['type'] == "energy":
                database['conditioning_features']['0']['conditioning_0_step_6_energy'].append((data_temp[i]['features'][2]['value']))
        if data_temp[i]['substep'] == "cycle_1_step_3":
            if data_temp[i]['features'][0]['type'] == "start_voltage":
                database['conditioning_features']['1']['conditioning_1_step_3_start_voltage'].append((data_temp[i]['features'][0]['value']))
            if data_temp[i]['features'][1]['type'] == "duration":
                database['conditioning_features']['1']['conditioning_1_step_3_duration'].append((data_temp[i]['features'][1]['value']))
            if data_temp[i]['features'][2]['type'] == "energy":
                database['conditioning_features']['1']['conditioning_1_step_3_energy'].append((data_temp[i]['features'][2]['value']))
            if data_temp[i]['features'][3]['type'] == "r_dc_step":
                database['conditioning_features']['1']['conditioning_1_step_3_r_dc_step'].append((data_temp[i]['features'][3]['value']))
            if data_temp[i]['features'][4]['type'] == "r_dc":
                database['conditioning_features']['1']['conditioning_1_step_3_r_dc'].append((data_temp[i]['features'][4]['value']))
        if data_temp[i]['substep'] == "cycle_1_step_4":
            if data_temp[i]['features'][0]['type'] == "start_voltage":
                database['conditioning_features']['1']['conditioning_1_step_4_start_voltage'].append((data_temp[i]['features'][0]['value']))
            if data_temp[i]['features'][1]['type'] == "duration":
                database['conditioning_features']['1']['conditioning_1_step_4_duration'].append((data_temp[i]['features'][1]['value']))
            if data_temp[i]['features'][2]['type'] == "energy":
                database['conditioning_features']['1']['conditioning_1_step_4_energy'].append((data_temp[i]['features'][2]['value']))
        if data_temp[i]['substep'] == "cycle_1_step_5":
            if data_temp[i]['features'][0]['type'] == "start_voltage":
                database['conditioning_features']['1']['conditioning_1_step_5_start_voltage'].append((data_temp[i]['features'][0]['value']))
            if data_temp[i]['features'][1]['type'] == "duration":
                database['conditioning_features']['1']['conditioning_1_step_5_duration'].append((data_temp[i]['features'][1]['value']))
            if data_temp[i]['features'][2]['type'] == "energy":
                database['conditioning_features']['1']['conditioning_1_step_5_energy'].append((data_temp[i]['features'][2]['value']))
        if data_temp[i]['substep'] == "cycle_1_step_6":
            if data_temp[i]['features'][0]['type'] == "start_voltage":
                database['conditioning_features']['1']['conditioning_1_step_6_start_voltage'].append((data_temp[i]['features'][0]['value']))
            if data_temp[i]['features'][1]['type'] == "duration":
                database['conditioning_features']['1']['conditioning_1_step_6_duration'].append((data_temp[i]['features'][1]['value']))
            if data_temp[i]['features'][2]['type'] == "energy":
                database['conditioning_features']['1']['conditioning_1_step_6_energy'].append((data_temp[i]['features'][2]['value']))


data_0 = pd.DataFrame(database['conditioning_features']['0'])
data_1 = pd.DataFrame(database['conditioning_features']['1'])

data = pd.concat([data_0, data_1], axis=1, join='inner')

# get all features and store them into dataframe

data['index'] = index
data = data.sort_values(by = ['index'])
data = data.drop(['index'], axis = 1)

labels = pd.read_csv("app/data/4_raw_q/labels.csv", delimiter=";")
labels = (labels.iloc[:, 1]).tolist()
data['qdot_labels'] = labels

# apply pca
scaler = StandardScaler()
segmentation_std = scaler.fit_transform(data)

pca = PCA(n_components=3)
pca.fit(segmentation_std)
scores_pca = pca.transform(segmentation_std)

data_pca = pd.concat([data.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
data_pca.columns.values[-3: ] = ['Component 1', 'Component 2', 'Component 3']
data = data_pca[['Component 1', 'Component 2', 'Component 3']]

# apply kmeans
kmeans_pca = KMeans(n_clusters=6, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)
data['Segment K-means PCA'] = kmeans_pca.labels_

# plot pca-kmeans class
data['Segment'] = data['Segment K-means PCA'].map({0:'first',1:'second',2:'third',3:'fourth',4:'fifth',5:'sixth'})
x_axis = data['Component 2']
y_axis = data['Component 1']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis, y_axis, hue = data['Segment'], palette = ['g','r','c','m','b','purple'])
plt.title('Clusters by PCA Components')
plt.show()

# generate csv
crino = {'index': index, 'label': kmeans_pca.labels_}
crino = pd.DataFrame(crino, columns = ['index','label'])
crino = crino.sort_values(by = ['index'])
crino.to_csv('Conditioning_PCA_Kmeans_label.csv') 

label = crino['label'].tolist()
class_0, class_1, class_2, class_3, class_4, class_5 = [], [], [], [], [], []

# print different classes
for i in range(len(label)):
    if label[i] == 0:
        class_0.append(index[i])
    if label[i] == 1:
        class_1.append(index[i])
    if label[i] == 2:
        class_2.append(index[i])
    if label[i] == 3:
        class_3.append(index[i])
    if label[i] == 4:
        class_4.append(index[i])
    if label[i] == 5:
        class_5.append(index[i])

print('class_0:', class_0)
print('\n')
print('class_1:',class_1)
print('\n')
print('class_2:',class_2)
print('\n')
print('class_3:',class_3)
print('\n')
print('class_4:',class_4)
print('\n')