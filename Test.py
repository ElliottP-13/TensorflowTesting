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

# Utilities
import pickle

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.keys = ['loss', 'acc', 'val_loss', 'val_acc']
        self.values = {}
        for k in self.keys:
            self.values['batch_'+k] = []
            self.values['epoch_'+k] = []

    def on_batch_end(self, batch, logs={}):
        for k in self.keys:
            bk = 'batch_'+k
            if k in logs:
                self.values[bk].append(logs[k])

    def on_epoch_end(self, epoch, logs={}):
        for k in self.keys:
            ek = 'epoch_'+k
            if k in logs:
                self.values[ek].append(logs[k])

    def plot(self, keys):
        for key in keys:
            plt.plot(np.arange(len(self.values[key])), np.array(self.values[key]), label=key)
        plt.legend()

def run_keras(X_train, y_train, X_val, y_val, X_test, y_test, layers, epochs, split=0, verbose=True):
    # Model specification
    model = Sequential()
    for layer in layers:
        model.add(layer)
    # Define the optimization
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=["accuracy"])
    N = X_train.shape[0]
    # Pick batch size
    batch = 32 if N > 1000 else 1     # batch size
    history = LossHistory()
    # Fit the model
    if X_val is None:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_split=split,callbacks=[history], verbose=verbose)
    else:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_val, y_val),callbacks=[history], verbose=verbose)
    # Evaluate the model on validation data, if any
    if X_val is not None or split > 0:
        val_acc, val_loss = history.values['epoch_val_acc'][-1], history.values['epoch_val_loss'][-1]
        print ("\nLoss on validation set:"  + str(val_loss) + " Accuracy on validation set: " + str(val_acc))
    else:
        val_acc = None
    # Evaluate the model on test data, if any
    if X_test is not None:
        test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch)
        print ("\nLoss on test set:"  + str(test_loss) + " Accuracy on test set: " + str(test_acc))
    else:
        test_acc = None
    return model, history, val_acc, test_acc


def keras_trials(trials, X_train, Y_train, X_test, Y_test, layers,split=.1):
    val_acc = 0
    test_acc = 0
    for trial in range(trials):
        # Reset the weights
        # See https://github.com/keras-team/keras/issues/341
        session = K.get_session()
        for layer in layers:
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if hasattr(v_arg, 'initializer'):
                    initializer_func = getattr(v_arg, 'initializer')
                    initializer_func.run(session=session)
        # Run the model
        model, history, vacc, tacc = run_keras(X_train,Y_train,None,None,X_test,Y_test,layers,20,split,verbose=False)
        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0

    #print("Average test accuracy: ",test_acc/trials)
    return model,test_acc/trials

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
layers = [Dense(input_dim=n, units=20, activation='relu'),
                Dense(units=10, activation='relu'),
                Dense(units=1, activation="linear")]

model, acc = keras_trials(2, x, y, x, y, layers)


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


