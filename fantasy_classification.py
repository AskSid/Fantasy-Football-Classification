import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
sc = StandardScaler()


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


df = pd.read_csv('QBdata.csv')
#df = pd.read_csv('RBdata.csv')
#df = pd.read_csv('WRdata.csv')
#df = pd.read_csv('TEdata.csv')


for i in df:
    if (i != 'Player') and (i != 'Draft') and (i != 'Tm') and (i != 'Lg'):
        df[i] = pd.to_numeric(df[i], errors='coerce')

df_2018 = df.loc[df['Year'] == 2018]
df_2018 = df_2018.dropna()
df_2018_adjusted = sc.fit_transform(df_2018[['Age', 'G', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Att.1', 'Yds.1', 'TD.1', 'Rec', 'Yds.2', 'TD.2', 'Fmb']])
df_2018_player = df_2018[['Player', 'PPR']]
df = df.loc[df['G'] >= 8]
df = df.loc[df['Year'] != 2018]
df = df.loc[df['G'] >= 8]

df = df[['Player', 'PPR','Age', 'G', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Att.1', 'Yds.1', 'TD.1', 'Rec', 'Yds.2', 'TD.2', 'Fmb']]
df = df.sort_values(['Player', 'Age'], ascending = [True, False])

grouped = df.groupby(df.Player)
frames = []
for name, group in grouped:
    group.sort_values(['Age'], ascending = [False])
    group = group.shift(1)
    frames.append(group)
df = pd.concat(frames)
df = df.dropna()
size = df.shape[0]


X = df[['Age', 'G', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Att.1', 'Yds.1', 'TD.1', 'Rec', 'Yds.2', 'TD.2', 'Fmb']]

X = sc.fit_transform(X)

df['PPR'] = df['PPR'].fillna(1)
Y = df['PPR'] / df['G']

a = 0
Y_new = np.zeros(size)
for i in Y:
    if i > 25:
        Y_new[a] = 5
    elif (i < 25.0) and (i >= 20.0):
        Y_new[a] = 4
    elif (i < 20.0) and (i >= 15.0):
        Y_new[a] = 3
    elif (i < 15.0) and (i >= 10.0):
        Y_new[a] = 2
    else:
        Y_new[a] = 1
    a += 1

Y = Y_new 
Y = np.asanyarray(Y)
Y = Y.reshape(-1, 1)
ohe = OneHotEncoder()
Y = ohe.fit_transform(Y).toarray()

Y[np.isnan(Y)] = 0
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1)


model = Sequential()
model.add(Dense(14, input_dim = 14, activation = 'relu'))
model.add(Dense(14, activation = 'relu'))
model.add(Dense(14, activation = 'relu'))
model.add(Dense(14, activation = 'relu'))
model.add(Dense(Y.shape[1], activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


history = model.fit(X_train, Y_train, batch_size = 20, verbose=1, epochs = 100, validation_split=0.2)

Y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(Y_pred)):
    pred.append(np.argmax(Y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(Y_test)):
    test.append(np.argmax(Y_test[i]))
ac = accuracy_score(pred,test)
print('Accuracy is:', ac*100)

model.evaluate(X_test, Y_test)
to_predict = X_train[:]
predictions = model.predict(df_2018_adjusted)

pred = []
df_results = df_2018[['Player']]
print(df_results.size)
print(predictions.shape)

for i in predictions:
    pred.append(np.where(i == np.amax(i))[0][0] + 1)

df_results['Predictions'] = pred
df_results = df_results.sort_values(['Predictions', 'Player'], ascending = [False, True])
df_results.head(30)
