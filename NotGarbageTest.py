import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.datasets import mnist
from keras import backend as K
from keras.initializers import VarianceScaling

import matplotlib as plt

def time_to_sec(time):

    time = str(time)
    splice = time.split(":")

    for i in range(len(splice)):
        splice[i] = int(splice[i])

    hour = splice[0]
    minutes = splice[1]
    sec = splice[2]

    return (hour * 60 * 60 + minutes * 60 + sec)

data = pd.read_excel('insulin_data1.xlsx', sheet_name='data', converters={'Time':time_to_sec})

time = data['Time']
carbs = data['CarbSize']
insulin = data['ActualTotalBolusRequested']
current_bg = data['Adjusted BG']
bg = data['BG in 2 hrs']
X = data[:, 0:-1]
print(X)
load_x = []
load_y = []

for i in range(len(insulin)):
    data_point = np.array([time[i], carbs[i], current_bg[i], bg[i]])
    load_x.append(data_point)
    load_y.append(insulin[i])

x = np.array(load_x)
y = np.array(load_y)

d,n = x.shape

print(d)
print(n)
print(x[0])
print(y[0])
print(len(x))
print(len(y))


print("************************************************************************************")

# create model
model = Sequential()
model.add(Dense(100, input_dim=n, activation='relu'))
model.add(Dense(1, activation='linear'))


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=500, batch_size=5)


#How to use the model to predict on input data
#Time, Carbs, Current BG, BG in 2Hrs

extra_test = np.array([[time_to_sec("09:15:13"), 80, 20, 125]])
print("predicted insulin delivery: ", model.predict(extra_test))

extra_test = np.array([[time_to_sec("09:15:13"), 80, 124, 125]])
print("predicted insulin delivery: ", model.predict(extra_test))

extra_test = np.array([[time_to_sec("09:15:13"), 80, 234, 125]])
print("predicted insulin delivery: ", model.predict(extra_test))

extra_test = np.array([[time_to_sec("09:15:13"), 80, 334, 125]])
print("predicted insulin delivery: ", model.predict(extra_test))


