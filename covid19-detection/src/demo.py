#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
import random
from os import listdir
from os.path import isfile, join
import keras.callbacks as kcall


# In[ ]:


train_dir = '../input/train/'
test_dir = '../input/test/'

labels = ['pneumonia', 'COVID-19', 'normal']
image_shape = (1024, 1024)
color_mode = 'grayscale'
batch_size = 16
epochs = 3


# In[ ]:


datagen = ImageDataGenerator(
                    samplewise_center=True,
                    samplewise_std_normalization=True
                    )

train_generator = datagen.flow_from_directory(train_dir, target_size = image_shape, batch_size = batch_size, color_mode = color_mode)
test_generator = datagen.flow_from_directory(test_dir, target_size = image_shape, batch_size = batch_size, color_mode = color_mode)
val_generator = datagen.flow_from_directory(val_dir, target_size = image_shape, batch_size = batch_size, color_mode = color_mode)

train_size = 12092
test_size = 1509
val_size = 1510


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
model = Sequential()
model.add(Input((1024,1024,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


# ## Callbacks

# In[ ]:


# early_stop = kcall.EarlyStopping(monitor = 'acc', min_delta=0.0001)
# tensorboard =kcall.TensorBoard(log_dir='./tensorboard-logs',write_grads=1,batch_size = batch_size)

# class LossHistory(kcall.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []
#         self.acc = []

#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.acc.append(logs.get('acc'))

# history = LossHistory()


# In[ ]:


history = model.fit(train_generator,
        steps_per_epoch = train_size // batch_size,
        epochs = epochs,
        validation_data = val_generator)


# In[ ]:


# Plot Model's Train v/s Validation Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Acuracy")
plt.legend(['Train', 'Validation'])
plt.show()

# Plot Model's Train v/s Validation Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train', 'Validation'])
plt.show()


# In[ ]:


#Evaluate the model's perfomance
performance = model.evaluate_generator(test_generator)
print("Loss on Test Set: %.2f" % (performance[0]))
print("Accuracy on Test Set: %.2f" % (performance[1]*100))

