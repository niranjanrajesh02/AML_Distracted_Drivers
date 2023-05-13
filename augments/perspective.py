import cv2
import numpy as np








def apply_random_perspective_transform(image):
    shift_range = 10
    scale_range = 0.1
    height, width, _ = image.shape

    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst = np.float32([[np.random.uniform(-shift_range, shift_range), np.random.uniform(-shift_range, shift_range)],
                      [width + np.random.uniform(-shift_range, shift_range), np.random.uniform(-shift_range, shift_range)],
                      [width + np.random.uniform(-shift_range, shift_range), height + np.random.uniform(-shift_range, shift_range)],
                      [np.random.uniform(-shift_range, shift_range), height + np.random.uniform(-shift_range, shift_range)]])

   
    dst += np.random.randn(4, 2) * scale_range * np.array([[width, height]])

   
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped





if __name__ == "__main__":
    image = cv2.imread('C:/Niranjan/Ashoka/Semester 6/AML/FinalProject/Data/imgs/train/c0/img_34.jpg')
    image = cv2.resize(image, (224, 224))
    cv2.imshow("Original", image)
    cv2.imshow("Warped", apply_random_perspective_transform(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()