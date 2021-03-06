# load the model from disk
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.svm import LinearSVC
from skimage import feature, exposure, transform
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob

num = 82

detections = []


def main():
    loaded_model = joblib.load('../lab/airplane_model')
    tp = 0
    fp = 0
    scale = 0
    for i in range(1):
        filename = '../test/devos/1.bmp'
        img = cv2.imread(filename)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_image = np.sqrt(gray_image / np.max(gray_image))  # gamma校正

        h, w = gray_image.shape
        cx, cy = w // 2, h // 2
        x, y = np.meshgrid(np.arange(w) - cx, np.arange(h) - cy)  # 生成一個座標矩陣
        circular_patch = gray_image.copy()
        circular_patch[x ** 2 + y ** 2 >= (min(cx, cy)) ** 2] = 0

        for i, row in enumerate(sliding_window_view(circular_patch, (64, 64))[::10, ::10]):
            for j, col in enumerate(row):
                (fd, hog_image) = feature.hog(col, orientations=9, pixels_per_cell=(8, 8),
                                              cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys',
                                              visualize=True)
                fds = fd.reshape(1, -1)
                pred = loaded_model.predict(fds)[0]
                pred.sum()
                if pred == 1:
                    x = j * 10
                    y = i * 10
                    w = 64
                    h = 64
                    if loaded_model.decision_function(
                            fds) > 0.7:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
                        print("Detection:: Location -> ({}, {})".format(x, y))
                        print(
                            "Scale ->  {} | Confidence Score {} \n".format(scale, loaded_model.decision_function(fds)))
                        detections.append((x, y, w, h))
        scale += 1

    for (x, y, w, h) in detections:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    cv2.imshow("Raw Detections after NMS", img)
    cv2.waitKey(0) & 0xFF
    # print(f'False Positive: {(tp / num) * 100} %')
    # print(f'False negative: {(fp / num) * 100} %')


if __name__ == '__main__':
    main()
