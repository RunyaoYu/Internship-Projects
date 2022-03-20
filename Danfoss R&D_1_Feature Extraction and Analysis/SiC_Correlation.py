# -*- coding: utf-8 -*-

import pandas as pd
from scipy.integrate import simps
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r'C:\Users\U388316\OneDrive - Danfoss\Desktop\feature\rivian.csv')
data = data.drop(data.columns[0], axis=1) #axis=1 refers to colomn. here delete the first column
class_le = LabelEncoder()

def correlation_matrix(dataframe, th=0.5):
    
    corr_mat2 = dataframe.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(15,12))         # Sample figsize in inches
    sns.heatmap(corr_mat2, annot=True,linewidths=.5, ax=ax )
    plt.show()

    corr_pairs = corr_mat2.unstack()
    sorted_pairs = corr_pairs.sort_values(kind="quicksort")
    strong_pairs = sorted_pairs[abs(sorted_pairs) > th]
    print('\n')
    print('High Correlation:')
    print(strong_pairs)
    
    
correlation_matrix(data, th=0.5)

