import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow import keras 
from model import create_model
from utils import plot_loss, plot_accuracy
from keras.utils import image_dataset_from_directory
from keras.callbacks import EarlyStopping
DATA_PATH = '/storage/niranjan.rajesh_ug23/aml/hybrid'

model = create_model()
model.compile(loss='categorical_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])

train_data = image_dataset_from_directory(os.path.join(DATA_PATH, 'train'), labels='inferred', label_mode="categorical", seed=42,
                                          validation_split=0.2, subset='training')
valid_data = image_dataset_from_directory(os.path.join(DATA_PATH, 'train'), labels='inferred', label_mode="categorical", seed=42,
                                          validation_split=0.2, subset='validation')


def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply normalization to the dataset
train_data = train_data.map(preprocess)
valid_data = valid_data.map(preprocess)

batch_size = 40

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
callbacks = [es]

history = model.fit(train_data, validation_data=valid_data, epochs=20, batch_size=batch_size, verbose=1 , callbacks=callbacks)

plot_loss(history)
plot_accuracy(history)
model.save("/home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/models/augtrain_vanilla2.h5")