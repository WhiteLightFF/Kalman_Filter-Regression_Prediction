from numpy import array
from numpy import hstack
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import seaborn as sns

SourceSimple = pd.read_csv('D:/LETI/Course 6/Diplom/Compile_Calman+Tracker/Source.csv')
Source = pd.read_csv('D:/LETI/Course 6/Diplom/Compile_Calman+Tracker/SourceHard.csv')
Predict = pd.DataFrame(np.zeros(0, dtype=[('x0', 'i8'), ('y0', 'i8')]))


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
in_seq1 = np.array(Source['x0']) / 1920  # ([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array(Source['y0']) / 1080  # ([15, 25, 35, 45, 55, 65, 75, 85, 95])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 3
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model

train = 1


model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # fit model
if(train):
    history = model.fit(X, y, epochs=100, verbose=1, validation_split=0.2)
# demonstrate prediction


# for i in range(0, len(SourceSimple), n_steps_out):
# 	input_coord = np.array(SourceSimple.loc[i:i + 2])
# 	x_in = (input_coord[:, 0] / 1920).reshape((n_steps_out, 1))
# 	y_in = (input_coord[:, 1] / 1080).reshape((n_steps_out, 1))
# 	xy_in = hstack((x_in, y_in)).reshape((1, n_steps_in, n_features))
#
# 	yhat = model.predict(xy_in, verbose=0)
#
# 	yhat = yhat.reshape((3, 2))
# 	x_pred = (yhat[:, 0] * 1920).reshape((n_steps_out, 1))
# 	y_pred = (yhat[:, 1] * 1080).reshape((n_steps_out, 1))
# 	xy_pred = hstack((x_pred, y_pred))
# 	adding = pd.DataFrame(xy_pred, columns=['x0', 'y0'])
# 	Predict = Predict.append(adding, ignore_index=True)
class LSTM:
    pointX = deque([0, 0, 0], maxlen=3)
    pointY = deque([0, 0, 0], maxlen=3)
    n_steps_out = 3

    def LSTM_Predict(self, x, y):
        self.pointX.append(x/1920)
        self.pointY.append(y/1080)
        x_in = np.array(self.pointX).reshape((n_steps_out, 1))
        y_in = np.array(self.pointY).reshape((n_steps_out, 1))
        xy_in = hstack((x_in, y_in)).reshape((1, n_steps_in, n_features))

        yhat = model.predict(xy_in, verbose=0)

        yhat = yhat.reshape((3, 2))
        x_pred = (yhat[:, 0] * 1920).reshape((n_steps_out, 1))
        y_pred = (yhat[:, 1] * 1080).reshape((n_steps_out, 1))
        xy_pred = hstack((x_pred, y_pred))
        return xy_pred
