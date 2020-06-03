#!/usr/bin/env python
# coding: utf-8

# # Ocular Disease Recognition

# ## Imports

# In[1]:


import numpy as np
import xlrd
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import random
from os import listdir
from os.path import isfile, join


# ## Constants

# In[ ]:





# ## load dataset

# In[2]:


train_path = '../input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images/'
test_path = '../input/ocular-disease-recognition-odir5k/ODIR-5K/Testing Images/'
train_images = [f for f in listdir(train_path) if isfile(join(train_path, f))]
test_images = [f for f in listdir(test_path) if isfile(join(test_path, f))]
print(len(train_images), len(test_images))
df = pd.read_excel('../input/ocular-disease-recognition-odir5k/ODIR-5K/data.xlsx')
print(df.head())


# ## Model

# In[3]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1000, 1000, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


# In[ ]:




