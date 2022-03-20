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

query_1 = {"$and": [{"type": "crino_eol"}, {"substep": "qdot"}, {"features": {"$exists": True}}]}
query_1 = list(collection.find(query_1))
database['features']['qdot_after_24h'] = []
database['features']['qdot_continuity'] = []

for i in range(len(query_1)):
    database['features']['qdot_after_24h'].append(query_1[i]['features'][0]['value'])
    database['features']['qdot_continuity'].append(query_1[i]['features'][1]['value'])
    index.append(query_1[i]['cell_id'])

data = pd.DataFrame(database['features'])

class_le = LabelEncoder()
for column in data[['qdot_continuity']].columns:
    data[column] = class_le.fit_transform(data[column].values)

# get all features and store them into dataframe

data['index'] = index
data = data.sort_values(by = ['index'])
data = data.drop(['index'], axis = 1)

# apply pca
scaler = StandardScaler()
segmentation_std = scaler.fit_transform(data)

pca = PCA(n_components=2)
pca.fit(segmentation_std)
scores_pca = pca.transform(segmentation_std)

data_pca = pd.concat([data.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
data_pca.columns.values[-2: ] = ['Component 1', 'Component 2']
data = data_pca[['Component 1', 'Component 2']]

# apply kmeans
kmeans_pca = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)
data['Segment K-means PCA'] = kmeans_pca.labels_

# plot pca-kmeans class
data['Segment'] = data['Segment K-means PCA'].map({0:'first',1:'second',2:'third',3:'fourth',4:'fifth'})
x_axis = data['Component 2']
y_axis = data['Component 1']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis, y_axis, hue = data['Segment'], palette = ['g','r','c','m','y'])
plt.title('Clusters by PCA Components')
plt.show()

# generate csv
crino = {'index': index, 'label': kmeans_pca.labels_}
crino = pd.DataFrame(crino, columns = ['index','label'])
crino = crino.sort_values(by = ['index'])
crino.to_csv('Raw_q_PCA_Kmeans_label.csv') 

# print different classes
label = crino['label'].tolist()
class_0, class_1, class_2, class_3, class_4 = [], [], [], [], []

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

print('class_0:',class_0)
print('\n')
print('class_1:',class_1)
print('\n')
print('class_2:',class_2)
print('\n')
print('class_3:',class_3)
print('\n')
print('class_4:',class_4)