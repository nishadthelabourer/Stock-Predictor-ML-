#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[2]:


data = pd.read_csv('C:/Users/admin/Videos/Equity-INFY-EQ-03-01-2019-to-03-01-2020.csv', parse_dates=[0], index_col=[0])


# In[3]:


df = data[['Close']]


# In[4]:


data_training = df.loc['2019-10-30':'2019-01-03',:]['Close']
data_test = df.loc['2020-01-03':'2019-10-31',:]['Close']


# In[5]:


x_train = []
y_train = []


# In[6]:


for i in range(5, len(data_training)-5):
    x_train.append(data_training[i-5:i])
    y_train.append(data_training[i:i+5])


# In[7]:


x_test = []
y_test = []


# In[8]:


for i in range(5, len(data_test)-5):
    x_test.append(data_test[i-5:i])
    y_test.append(data_test[i:i+5])


# In[9]:


x_train , y_train = np.array(x_train), np.array(y_train)
x_test , y_test = np.array(x_test), np.array(y_test)


# In[10]:


x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
x_train = x_scaler.fit_transform(x_train)
y_train = y_scaler.fit_transform(y_train)
x_test = x_scaler.fit_transform(x_test)
y_test = y_scaler.fit_transform(y_test)


# In[11]:


x_train = x_train.reshape(191,5,1)


# In[12]:


reg = Sequential()

reg.add(LSTM(units=150, activation='relu', return_sequences=True, input_shape=(5,1)))

reg.add(LSTM(units=41, activation='relu'))
reg.add(Dropout(0.2))

reg.add(Dense(5))


# In[13]:


reg.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# In[14]:


reg.fit(x_train, y_train, epochs=200, batch_size=5)


# In[18]:


x_test = x_test.reshape(35,5,1)


# In[19]:


y_pred = reg.predict(x_test)


# In[20]:


y_test = y_scaler.inverse_transform(y_test)


# In[21]:


y_pred.shape, y_test.shape


# In[22]:


y_pred = y_scaler.inverse_transform(y_pred)


# In[23]:


from sklearn.metrics import mean_squared_error 


# In[24]:


def evaluating_model(y_test,y_pred):
    scores=[]
    
    for i in range(y_test.shape[1]):
        mse = mean_squared_error(y_test[:,i], y_pred[:,i])
        rmse = np.sqrt(mse)
        scores.append(rmse)
        
    return scores


# In[25]:


evaluating_model(y_test,y_pred)


# In[28]:


np.std(y_test[0]), np.std(y_test[1]), np.std(y_test[2])


# In[31]:


pred = y_pred.reshape(175,1)
test = y_test.reshape(175,1)


# In[37]:


plt.figure(figsize=(14,5))
plt.plot(pred, color='red', label='Predicted')
plt.plot(test, color='blue', label='True Values')
plt.legend()
plt.ylabel(' Stock Price INFY')
plt.title('INFY Stock Price VS ML Prediction')
plt.grid()


# In[ ]:




