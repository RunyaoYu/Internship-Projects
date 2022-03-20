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

database = {}
database['features'] = {}
index = []

query_1 = {"$and": [{"type": "formation"}, {"substep": "soacking"}, {"features": {"$exists": True}}]}
query_1 = list(collection.find(query_1))
database['features']['soacking_state'] = []
database['features']['soacking_success'] = []
database['features']['soacking_start_voltage'] = []
for i in range(len(query_1)):
    database['features']['soacking_state'].append(query_1[i]['features'][0]['value'])
    database['features']['soacking_success'].append(query_1[i]['features'][1]['value'])
    database['features']['soacking_start_voltage'].append(query_1[i]['features'][2]['value'])
    index.append(query_1[i]['cell_id'])

query_2 = {"$and": [{"type": "formation"}, {"substep": "precharge"}, {"features": {"$exists": True}}]}
query_2 = list(collection.find(query_2))
database['features']['precharge_start_voltage'] = []
database['features']['precharge_duration'] = []
database['features']['precharge_energy'] = []
database['features']['precharge_success'] = []
database['features']['precharge_continuity'] = []
for i in range(len(query_2)):
    database['features']['precharge_start_voltage'].append(query_2[i]['features'][0]['value'])
    database['features']['precharge_duration'].append(query_2[i]['features'][1]['value'])
    database['features']['precharge_energy'].append(query_2[i]['features'][2]['value'])
    database['features']['precharge_success'].append(query_2[i]['features'][3]['value'])
    database['features']['precharge_continuity'].append(query_2[i]['features'][4]['value'])

query_3 = {"$and": [{"type": "formation"}, {"substep": "charge"}, {"features": {"$exists": True}}]}
query_3 = list(collection.find(query_3))
database['features']['charge_start_voltage'] = []
database['features']['charge_duration'] = []
database['features']['charge_energy'] = []
database['features']['charge_success'] = []
database['features']['charge_continuity'] = []
for i in range(len(query_3)):
    database['features']['charge_start_voltage'].append(query_3[i]['features'][0]['value'])
    database['features']['charge_duration'].append(query_3[i]['features'][1]['value'])
    database['features']['charge_energy'].append(query_3[i]['features'][2]['value'])
    database['features']['charge_success'].append(query_3[i]['features'][3]['value'])
    database['features']['charge_continuity'].append(query_3[i]['features'][4]['value'])

query_4 = {"$and": [{"type": "formation"}, {"substep": "discharge"}, {"features": {"$exists": True}}]}
query_4 = list(collection.find(query_4))
database['features']['discharge_start_voltage'] = []
database['features']['discharge_duration'] = []
database['features']['discharge_energy'] = []
database['features']['discharge_success'] = []
database['features']['discharge_continuity'] = []
for i in range(len(query_4)):
    database['features']['discharge_start_voltage'].append(query_4[i]['features'][0]['value'])
    database['features']['discharge_duration'].append(query_4[i]['features'][1]['value'])
    database['features']['discharge_energy'].append(query_4[i]['features'][2]['value'])
    database['features']['discharge_success'].append(query_4[i]['features'][3]['value'])
    database['features']['discharge_continuity'].append(query_4[i]['features'][4]['value'])

data = pd.DataFrame(database['features'])

class_le = LabelEncoder()
for column in data[['soacking_state', 'precharge_continuity', 'charge_continuity', 'discharge_continuity', 'soacking_success', 'precharge_success', 'charge_success', 'discharge_success']].columns:
    data[column] = class_le.fit_transform(data[column].values)

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

pca = PCA(n_components=6)
pca.fit(segmentation_std)
scores_pca = pca.transform(segmentation_std)

data_pca = pd.concat([data.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
data_pca.columns.values[-6: ] = ['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5', 'Component 6']
data = data_pca[['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5', 'Component 6']]

# apply kmeans
kmeans_pca = KMeans(n_clusters=8, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)
data['Segment K-means PCA'] = kmeans_pca.labels_

# plot pca-kmeans class
data['Segment'] = data['Segment K-means PCA'].map({0:'1st',1:'2nd',2:'3rd',3:'4th',4:'5th',5:'6th',6:'7th',7:'8th'})
x_axis = data['Component 2']
y_axis = data['Component 1']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis, y_axis, hue = data['Segment'], palette = ['g','purple','orange','brown','olive','gray','pink','gold'])
plt.title('Clusters by PCA Components')
plt.show()

# generate csv
crino = {'index': index, 'label': kmeans_pca.labels_}
crino = pd.DataFrame(crino, columns = ['index','label'])
crino = crino.sort_values(by = ['index'])
crino.to_csv('Formation_PCA_Kmeans_label.csv') 

label = crino['label'].tolist()
class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7 = [], [], [], [], [], [], [], []

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
    if label[i] == 6:
        class_6.append(index[i])
    if label[i] == 7:
        class_7.append(index[i])

print('class_0:',class_0)
print('\n')
print('class_1:',class_1)
print('\n')
print('class_2:',class_2)
print('\n')
print('class_3:',class_3)
print('\n')
print('class_4:',class_4)
print('\n')
print('class_5:',class_5)
print('\n')
print('class_6:',class_6)
print('\n')
print('class_7:',class_7)