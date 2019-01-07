import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_lstm_sequences(dataset, look_back=1):
  data_x, data_y = [], []

  for i in range(len(dataset)-look_back-1):
    upper_limit = i + look_back
    seq = dataset[i:upper_limit, 0]
    data_x.append(seq)
    data_y.append(dataset[i + look_back, 0])
  return np.array(data_x), np.array(data_y)

def create_lstm_net(lookback=1):
  model = Sequential()
  model.add(LSTM(4, input_shape=(1, look_back)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

# fix random seed
np.random.seed(2)

# get data from csv
df = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
data = df.values
data = data.astype('float32')

# normalize data

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# split train and test

train_size = int(len(data) * 0.7)
test_size = len(data) - train_size

train_data, test_data = data[0:train_size], data[train_size:]

# reshape X=t, Y=t+1
look_back = 5
train_x, train_y = create_lstm_sequences(train_data, look_back)
test_x, test_y = create_lstm_sequences(test_data, look_back)


# reshape input to be [samples, time steps, features]
train_x =  np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x =  np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

# train model
model = create_lstm_net(look_back)
plot_model(model, to_file='lstm.png')
model.fit(train_x, train_y, epochs=100, batch_size=1)

# make predictions

train_pred = model.predict(train_x)
test_pred = model.predict(test_x)

# invert pred to real values
train_pred = scaler.inverse_transform(train_pred)
train_y = scaler.inverse_transform([train_y])
test_pred = scaler.inverse_transform(test_pred)
test_y = scaler.inverse_transform([test_y])

# calculate RMSE
train_score = math.sqrt(mean_squared_error(train_y[0], train_pred[:,0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = math.sqrt(mean_squared_error(test_y[0], test_pred[:,0]))
print('Test Score: %.2f RMSE' % (test_score))


# shift train predictions for plotting
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_pred)+look_back, :] = train_pred
# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_pred)+(look_back*2)+1:len(data)-1, :] = test_pred
# plot baseline and predictions
plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

