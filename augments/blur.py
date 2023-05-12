import cv2
import numpy as np








def apply_random_gaussian_blur(image):
    kernel_size_range=(3, 7)
    kernel_size = np.random.choice(range(kernel_size_range[0], kernel_size_range[1] + 1, 2))
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return image


if __name__ == "__main__":
    image = cv2.imread('C:/Niranjan/Ashoka/Semester 6/AML/FinalProject/Data/imgs/train/c0/img_34.jpg')
    image = cv2.resize(image, (224, 224))
    cv2.imshow("Original", image)
    cv2.imshow("Brightened", apply_random_gaussian_blur(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()