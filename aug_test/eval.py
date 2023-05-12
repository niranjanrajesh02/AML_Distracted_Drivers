import tensorflow as tf
from tensorflow import keras
import os
from keras.utils import image_dataset_from_directory
import numpy as np

aug_data = "C:/Niranjan/Ashoka/Semester 6/AML/FinalProject/Data/aug"

test_model = tf.keras.models.load_model('C:/Niranjan/Ashoka/Semester 6/AML/FinalProject/AML_Distracted_Drivers/models/model_customcnn.h5', compile=False)
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