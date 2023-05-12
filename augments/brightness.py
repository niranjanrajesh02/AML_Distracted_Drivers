import cv2
import numpy as np








def apply_random_brightness(image):
    brightness_range=(0.5, 2.0)
    brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness_factor, 0, 255)
    b_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return b_img


if __name__ == "__main__":
    image = cv2.imread('C:/Niranjan/Ashoka/Semester 6/AML/FinalProject/Data/imgs/train/c0/img_34.jpg')
    image = cv2.resize(image, (224, 224))
    cv2.imshow("Original", image)
    cv2.imshow("Brightened", apply_random_brightness(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()