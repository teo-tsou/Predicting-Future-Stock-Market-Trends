# Recurrent Neural Network
# Theodoros Tsourdinis

#Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # Getting the second column of the file (Open) as a numpy array 

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1)) #We choose normalisation (for rnn) for the feature scaling instead of standardisation. The (0 and 1) is because all the new stock prices will be between zero and one
training_set_scaled = sc.fit_transform(training_set) #normalised scaling set

# Creating a data structure with 60 timesteps (past info before T time) and 1 output (T + 1)
x_train = []
y_train = []

for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i, 0]) #All the previous 60 values (0 - 59) each time in one column
    y_train.append(training_set_scaled[i,0]) #The next value after previous values (0-59 -> 60) each time in one column

# Make our columns as numpy arrays in order to feed our RNN
x_train = np.array(x_train)
y_train = np.array(y_train) 


# Reshaping - In order to add a new dimension (new indicators) in a numpy array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #3D array: 1d: number of observations , 2d: timesteps, 3d: extra dimension. For more info see keras docs.


# Part 2 - Building the RNN

# Initialising the RNN

regressor = Sequential() #Regression is about predicting the continous value. Here we are initialise the RNN 

# Adding  the first LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1 ))) #Here we add one LSTM layer: units: neurons-cells , return_seq: because we are gonna add another LSTM layer,  input_shape: includes all the dimensions except the first x_train.shape[0] (its taken already by default) 
regressor.add(Dropout(0.2)) #In order to do some regularisation. It takes the dropout rate. It is the rate of neurons that we wannt to drop.

# Adding another one
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) 

# Adding another one
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding another one -Last one
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))  

# Adding the Output Layer
regressor.add(Dense(units = 1)) #The output (prediction) has one dimension (one column). We used Dense class to fully connect the network.

# Compiling the RNN
regressor.compile(optimizer='adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(x_train, y_train, epochs= 100, batch_size = 32)

# Part 3 Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) # We need the 60 previous values, so we concatanate our test data with train data and then we will take as lower bound the first day of 2017 - 60 and as upper bound the last day of the test_data.
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values # We are adding values method to make it an numpy array
inputs = inputs.reshape(-1,1) # We didn't use the iloc method from Pandas to get the inputs in the previous line of code. As a result it isn't shaped the right way (like Numpy Array). So we reshape it.
inputs = sc.transform(inputs) #Normalised scaling set without fit_transform (just trasform), because our sc object was already fitted to the training set. We only scale the inputs and not the actual test values.
x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0]) #All the previous 60 values (0 - 59) each time in one column
    

x_test = np.array(x_test) # Make our columns as numpy arrays in order to feed our RNN
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) #3D array: 1d: number of observations , 2d: timesteps, 3d: extra dimension. For more info see keras docs.
predicted_stock_price = regressor.predict(x_test) #Prediction
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #Inverse scale


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price (Early 2017)')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price (Early 2017)')
plt.title('Google Stock Price Prediction (Real vs Predicted)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()