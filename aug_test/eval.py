import tensorflow as tf
from tensorflow import keras
import os
from keras.utils import image_dataset_from_directory
import numpy as np

aug_data = "/storage/niranjan.rajesh_ug23/aml/aug"

test_model = tf.keras.models.load_model('/home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/Results/augtrain_vanilla.h5', compile=False)
test_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

augments = ['blur', 'brightness', 'occlusion','perspective']

for augment in augments:
    
    test_data = image_dataset_from_directory(os.path.join(aug_data, augment), labels='inferred', label_mode="categorical", image_size=(224,224))

    def preprocess(image, label):
        gray_img =  tf.reduce_mean(image, axis=-1, keepdims=True)
    
        return gray_img, label

    test_data = test_data.map(preprocess)

    loss, acc = test_model.evaluate(test_data, verbose=2)
    print("AUGMENTATION:", augment)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))