import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
import random
from os import listdir
from os.path import isfile, join, dirname
import keras.callbacks as kcall
from keras.applications import InceptionV3, DenseNet201, ResNet50V2
import tensorflow as tf


train_dir = '../input/train/'
test_dir = '../input/test/'
val_dir = '../input/val/'
labels = ['pneumonia', 'COVID-19', 'normal']
img_width, img_height, channels = 500, 500, 3
color_mode = 'rgb'
batch_size = 4
epochs = 1

def get_data():
    datagen = ImageDataGenerator(
                    samplewise_center=True,
                    samplewise_std_normalization=True
                    )

    train_generator = datagen.flow_from_directory(
        train_dir, 
        target_size=(img_width, img_height), 
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical')
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical')
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical')

    train_size = 13830
    test_size = 300
    val_size = 150
    return train_generator, test_generator, val_generator


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        except RuntimeError as e:
            print(e)

    train, test, val = get_data()
    inception = InceptionV3(
        include_top=True,
        weights=None,
        input_shape=(img_width, img_height, channels),
        classes=3
    )
    resNet = ResNet50V2(
        include_top=True,
        weights=None,
        input_shape=(img_width, img_height, channels),
        classes=3
    )

    denseNet = DenseNet201(
        include_top=True,
        weights=None,
        input_shape=(img_width, img_height, channels),
        classes=3
    )

    inception.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    resNet.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    denseNet.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])