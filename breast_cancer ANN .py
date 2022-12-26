#!/usr/bin/env python
# coding: utf-8

# In[111]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix
import seaborn as sns


# In[5]:


data = pd.read_csv('breast_cancer_data.csv')


# In[6]:


data.head()


# In[7]:


data.isna().sum()


# In[8]:


data.shape


# In[11]:


data.drop(columns = ['Unnamed: 32'], inplace = True)


# In[13]:


data.describe()


# In[15]:


data['diagnosis'].unique()


# In[67]:


data.head() # m = 1 and b = 0 


# In[46]:


data['diagnosis'] = pd.Categorical(data.diagnosis).codes
data['diagnosis'].unique()


# In[47]:


# feature and target variable  separation
X = data.drop(columns =['diagnosis'])
y = data['diagnosis']
print('X shape', X.shape)
print('y shape', y.shape)


# In[48]:


# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 42)


# In[49]:


print('X_train shape:', X_train.shape, '   X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape, '   y_test shape:', y_test.shape)


# In[55]:


# standard Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[116]:


model = keras.Sequential([
        keras.layers.Dense(30, input_shape = (31,), activation = 'relu'),
        keras.layers.Dense(10, activation = 'relu'),
        keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile( optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 20)


# In[60]:


model.evaluate(X_test, y_test)


# In[101]:


y_pred = pd.Series(model.predict(X_test).flatten())
y_pred = (y_pred > 0.5).astype(int)
y_pred[:5]


# In[114]:


cm = tf.math.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot= True)


# In[104]:


accuracy_score(y_test, y_pred)


# In[109]:


clf = classification_report(y_test, y_pred, output_dict = True)
sns.heatmap(pd.DataFrame(clf), annot= True)


# In[ ]:




