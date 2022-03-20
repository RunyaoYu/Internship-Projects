"""
Author: Runyao Yu
runyao.yu@tum.de
Research Internship in ETH Zurich
For Academic Use Purpose only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras import preprocessing
import os

path = "/content" # only change the path where the data stored together

def LSTM_Model(df):
    # 设置最频繁使用的5000个词/Set the number of most frequently used words
    MAX_NB_WORDS = 5000
    # 每条cut_review最大的长度/Max. length of your sentence
    MAX_SEQUENCE_LENGTH = 100
    # 设置Embeddingceng层的维度/Dim. of Embdedding Layer
    EMBEDDING_DIM = 100
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['text'].values)
    word_index = tokenizer.word_index

    X = tokenizer.texts_to_sequences(df['text'].values)
    #Padding let length be equal
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    Y = pd.get_dummies(df.iloc[:,1]).values

    #split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

    #shape to 2D tensor
    Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))

    #定义模型
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    #model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    #model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
    #print(model.summary())

    epochs = 10
    batch_size = 4
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.3,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    y_pred = model.predict(X_test)
    y_pred = y_pred.argmax(axis = 1)
    Y_test = Y_test.argmax(axis = 1)

    return accuracy_score(y_pred, Y_test)


def LSTM_Prediction(path):
    acc = []
    for root, dirs, files in os.walk(path): 
        for name in files:
            path = os.path.join(root, name)
            if "xlsx" in path:
                df = pd.read_excel(path) # one time for one excel, note the format is xlsx not csv
                result = LSTM_Model(df)
                print("Single accuracy for {}:".format(name),result)
                acc.append(result)
    print("Macro accuracy by using LSTM:",np.mean(acc))


LSTM_Prediction(path)