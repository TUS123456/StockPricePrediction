#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Specify the path to your CSV file
file_path = 'AAPL.csv'

# Use the read_csv function to read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame
print(df)


# In[2]:


df.head()


# In[3]:


df.tail()


# In[4]:


df1=df.reset_index()['close']


# In[5]:


df1


# In[6]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[7]:


import numpy as np


# In[8]:


df1


# In[9]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[10]:


print(df1)


# In[11]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[12]:


training_size,test_size


# In[13]:


train_data


# In[14]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[15]:


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[16]:


print(X_train.shape), print(y_train.shape)


# In[17]:


print(X_test.shape), print(ytest.shape)


# In[18]:


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[19]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[20]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[21]:


model.summary()


# In[22]:


model.summary()


# In[23]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[24]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[25]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[26]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[27]:


math.sqrt(mean_squared_error(ytest,test_predict))


# In[28]:


import numpy as np
import matplotlib.pyplot as plt

look_back = 100

trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict


testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict


plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[29]:


len(test_data)22/may------>23rd/may


# In[30]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[31]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[32]:


temp_input


# In[33]:


from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[34]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[35]:


import matplotlib.pyplot as plt


# In[36]:


plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[37]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[38]:


df3=scaler.inverse_transform(df3).tolist()


# In[39]:


plt.plot(df3)


# In[ ]:




