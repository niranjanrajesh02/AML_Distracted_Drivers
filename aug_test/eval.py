import tensorflow as tf
from tensorflow import keras
import os
from keras.utils import image_dataset_from_directory
import numpy as np

aug_data = "/storage/niranjan.rajesh_ug23/aml/aug"
model_path = '/home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/models'

models_to_eval = ['mobilenetmodel.h5', 'mobilenetmodel_pretrained.h5', 'mobile_customcnn.h5']

for model_name in models_to_eval:

    test_model = tf.keras.models.load_model(os.path.join(model_path, model_name), compile=False)
    test_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    augments = ['blur', 'brightness', 'occlusion','perspective', 'all']

    for augment in augments:
        
        if augment == 'all': data_path = os.path.join(aug_data, augment, 'test')
        else: data_path = os.path.join(aug_data, augment)
        
        test_data = image_dataset_from_directory(data_path, labels='inferred', label_mode="categorical", image_size=(224,224))


        loss, acc = test_model.evaluate(test_data, verbose=0)
        print("MODEL:", model_name, "AUGMENTATION:", augment)
        print("accuracy: {:5.2f}%".format(100 * acc))