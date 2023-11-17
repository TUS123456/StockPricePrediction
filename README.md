# StockPricePrediction

# Specify the path to your CSV file
file_path = 'AAPL.csv'

# Use the read_csv function to read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame
print(df)


### LSTM are sensitive to scale of the data. so we appl MinMax Scaler
​
### in this step I tranferred my data set value from in the range form (0,1)



### Now from here we convert data into train dataset and test dataset 
### for distibution of dataset  we multiple option for this because it is a time series dataset
### cross Validattion
### random seed

## for TimeSeries dataset we divide data on the basis of date because every index value depend upon on previous value

###  120,130,125,140,134,150|||| 160,190,154

### TimeSeries data set --->  Training dataset-120,130,125,140,134,150
#### test dataset -->160,190,154



### spillting data set perform
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]



## create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM



model = Sequential()

# Assuming your input data has a shape (number_of_samples, time_steps, features)
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))

# Stacked LSTM layers
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))

# Output layer
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
