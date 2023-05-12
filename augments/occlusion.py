import cv2
import numpy as np
import random




def apply_random_occlusion(img):
    height, width, channels = img.shape
    occlusion_height = random.randint(40, 80)
    occlusion_width = random.randint(40, 80)
    x = random.randint(0, width - occlusion_width)
    y = random.randint(0, height - occlusion_height)
    new_img = img.copy()
    new_img[y:y+occlusion_height, x:x+occlusion_width] = 0
    return new_img


if __name__ == "__main__":
    img = cv2.imread('C:/Niranjan/Ashoka/Semester 6/AML/FinalProject/Data/imgs/train/c0/img_34.jpg')
    img = cv2.resize(img, (224, 224))
    occluded_img = apply_random_occlusion(img)
    cv2.imshow("Original Image", img)
    cv2.imshow("Occluded Image", occluded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()