import numpy as np
import cv2
import os
import shutil
import random
from sklearn.model_selection import train_test_split
import argparse
from occlusion import apply_random_occlusion
from perspective import apply_random_perspective_transform
from brightness import apply_random_brightness
from blur import apply_random_gaussian_blur
from tqdm import tqdm
import numpy as np
from glob import glob
from keras.utils import np_utils

parser = argparse.ArgumentParser()

parser.add_argument("--augment", help="augmentation type", default="none")

args = parser.parse_args()



DATA_DIRECTORY = "C:/Niranjan/Ashoka/Semester 6/AML/FinalProject/Data/imgs/train/"
AUG_DIRECTORY = "C:/Niranjan/Ashoka/Semester 6/AML/FinalProject/Data/aug/"

def load_train(img_rows, img_cols, color_type=3, variant='data'):
    """
    Return train images and train labels from the original path
    """
    train_images = [] 
    train_labels = []
    # Loop over the training folder 
    for classes in range(10):
        print('Loading directory c{}'.format(classes))
        if variant == 'data':
            files = glob(DATA_DIRECTORY+ "c" + str(classes) + "/" +'*.jpg')
        elif variant == 'aug':
            files = glob(AUG_DIRECTORY+ "all/train/" "c" + str(classes) + "/" +'*.jpg')
        for file in files:
          train_images.append(file)
          train_labels.append(f'c{classes}')
    return train_images, train_labels 

def read_and_normalize_train_data(img_rows, img_cols, color_type, variant="data"):
    """
    Load + categorical + split
    """
    X, labels = load_train(img_rows, img_cols, color_type, variant=variant)
    # y = np_utils.to_categorical(labels, 10) #categorical train label
    X_train, X_valid, y_train, y_valid = train_test_split(X, labels, test_size=0.2, random_state=42) # split into train and test
    
    return X_train, X_valid, y_train, y_valid






if __name__ == "__main__":
    
    X_train, X_valid, y_train, y_valid = read_and_normalize_train_data(224, 224, 3)

    assert(len(X_valid) == len(y_valid))
    
    if (args.augment == 'all'):
        ALL_PATH = os.path.join(AUG_DIRECTORY, "all")
        
        if not os.path.exists(ALL_PATH):
            os.makedirs(ALL_PATH)
    
        for i in tqdm(range(len(X_train))):
            index = X_train[i].index("img_")
            f_name = X_train[i][index:]
            train_path = os.path.join(ALL_PATH, "train")
            if not os.path.exists(train_path):
                os.makedirs(train_path)
                
            img = cv2.imread(X_train[i])
            img = cv2.resize(img, (224, 224))
            label = y_train[i]
            class_path = os.path.join(train_path, label)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
                
            if i % 4 == 0:
                aug_img = apply_random_occlusion(img)
            elif i % 4 == 1:
                aug_img = apply_random_perspective_transform(img)
            elif i % 4 == 2:
                aug_img = apply_random_brightness(img)
            else:
                aug_img = apply_random_gaussian_blur(img)
                
            cv2.imwrite(os.path.join(class_path, f"{f_name}_aug.jpg"), aug_img)

        for i in tqdm(range(len(X_valid))):
            index = X_valid[i].index("img_")
            f_name = X_valid[i][index:]
            test_path = os.path.join(ALL_PATH, "test")
            if not os.path.exists(test_path):
                os.makedirs(test_path)
                
            img = cv2.imread(X_valid[i])
            img = cv2.resize(img, (224, 224))
            label = y_valid[i]
            class_path = os.path.join(test_path, label)
            
            if not os.path.exists(class_path):
                os.makedirs(class_path)
                
            if i % 4 == 0:
                aug_img = apply_random_occlusion(img)
            elif i % 4 == 1:
                aug_img = apply_random_perspective_transform(img)
            elif i % 4 == 2:
                aug_img = apply_random_brightness(img)
            else:
                aug_img = apply_random_gaussian_blur(img)
                
            cv2.imwrite(os.path.join(class_path, f"{f_name}_aug.jpg"), aug_img)

    if (args.augment == 'perspective'):
        PERS_PATH = os.path.join(AUG_DIRECTORY, "perspective")
        if not os.path.exists(PERS_PATH):
            os.makedirs(PERS_PATH)
        
        for i in tqdm(range(len(X_valid))):
            index = X_valid[i].index("img_")
            f_name = X_valid[i][index:]
            
            # print(f"Occluding {f_name}")
            
            img = cv2.imread(X_valid[i])
            img = cv2.resize(img, (224, 224))
            label = y_valid[i]
            class_path = os.path.join(PERS_PATH, label)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            pers_shift_img = apply_random_perspective_transform(img)
            cv2.imwrite(os.path.join(class_path, f"{f_name}_pers.jpg"), pers_shift_img)
            
    if (args.augment == "occlusion"):
        OCC_PATH = os.path.join(AUG_DIRECTORY, "occlusion")
        
        if not os.path.exists(OCC_PATH):
            os.makedirs(OCC_PATH)
        
        for i in tqdm(range(len(X_valid))):
            index = X_valid[i].index("img_")
            f_name = X_valid[i][index:]
            
            # print(f"Occluding {f_name}")
            img = cv2.imread(X_valid[i])
            img = cv2.resize(img, (224, 224))
            label = y_valid[i]
            class_path = os.path.join(OCC_PATH, label)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            occluded_img = apply_random_occlusion(img)
            cv2.imwrite(os.path.join(class_path, f"{f_name}_occ.jpg"), occluded_img)

    if (args.augment == "brightness"):
        BR_PATH = os.path.join(AUG_DIRECTORY, "brighntess")
        
        if not os.path.exists(BR_PATH):
            os.makedirs(BR_PATH)
        
        for i in tqdm(range(len(X_valid))):
            index = X_valid[i].index("img_")
            f_name = X_valid[i][index:]
            
            img = cv2.imread(X_valid[i])
            img = cv2.resize(img, (224, 224))
            label = y_valid[i]
            class_path = os.path.join(BR_PATH, label)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            brightened_img = apply_random_brightness(img)
            cv2.imwrite(os.path.join(class_path, f"{f_name}_br.jpg"), brightened_img)

    if (args.augment == "blur"):
        BL_PATH = os.path.join(AUG_DIRECTORY, "blur")
        
        if not os.path.exists(BL_PATH):
            os.makedirs(BL_PATH)
        
        for i in tqdm(range(len(X_valid))):
            index = X_valid[i].index("img_")
            f_name = X_valid[i][index:]
            
            img = cv2.imread(X_valid[i])
            img = cv2.resize(img, (224, 224))
            label = y_valid[i]
            class_path = os.path.join(BL_PATH, label)
            
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            
            blurred_img = apply_random_gaussian_blur(img)
            cv2.imwrite(os.path.join(class_path, f"{f_name}_bl.jpg"), blurred_img)
            
    if (args.augment == "hybrid"):
        HYBRID_PATH = os.path.join(AUG_DIRECTORY, "hybrid")
        
        aug_X_train, _, aug_y_train, _ = read_and_normalize_train_data(224, 224, 3, 'aug')
        
        
        if not os.path.exists(HYBRID_PATH):
            os.makedirs(HYBRID_PATH)
            
        for i in tqdm(range(len(X_train))):
            index = X_train[i].index("img_")
            f_name = X_train[i][index:]
            train_path = os.path.join(HYBRID_PATH, "train")
            if not os.path.exists(train_path):
                os.makedirs(train_path)
                
            img = cv2.imread(X_train[i])
            img = cv2.resize(img, (224, 224))
            label = y_train[i]
            class_path = os.path.join(train_path, label)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

            cv2.imwrite(os.path.join(class_path, f"{f_name}.jpg"), img)
            
        for i in tqdm(range(len(aug_X_train))):
            index = aug_X_train[i].index("img_")
            f_name = aug_X_train[i][index:]
            train_path = os.path.join(HYBRID_PATH, "train")
            if not os.path.exists(train_path):
                os.makedirs(train_path)
                
            img = cv2.imread(aug_X_train[i])
            img = cv2.resize(img, (224, 224))
            label = aug_y_train[i]
            class_path = os.path.join(train_path, label)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

            cv2.imwrite(os.path.join(class_path, f"{f_name}_aug.jpg"), img)