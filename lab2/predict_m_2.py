# load the model from disk
import numpy as np
from matplotlib import pyplot as plt
# from numpy.lib.stride_tricks import sliding_window_view
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



def main():
    count = 0
    # Define HOG Parameters
    # change them if necessary to orientations = 8, pixels per cell = (16,16), cells per block to (1,1) for weaker HOG
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    threshold = 1.2

    loaded_model = joblib.load('../lab/airplane_model')
    filename = '../test/devos/1.bmp'
    img = cv2.imread(filename)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = np.sqrt(gray_image / np.max(gray_image))  # gamma校正

    # Test the trained classifier on an image below!
    scale = 0
    detections = []

    # defining the size of the sliding window (has to be, same as the size of the image in the training data)
    (winW, winH) = (64, 64)
    windowSize = (winW, winH)
    downscale = 1.2
    # Apply sliding window:
    for resized in pyramid_gaussian(gray_image, downscale=1.5):  # loop over each layer of the image that you take!
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it!
            if window.shape[0] != winH or window.shape[
                1] != winW:  # ensure the sliding window has met the minimum size requirement
                continue
            fds = feature.hog(window, orientations, pixels_per_cell, cells_per_block,
                              block_norm='L2')  # extract HOG features from the window captured
            fds = fds.reshape(1, -1)  # re shape the image to make a silouhette of hog
            pred = loaded_model.predict(
                fds)  # use the SVM model to make a prediction on the HOG features extracted from the window

            if pred == 1:
                if  loaded_model.decision_function(
                        fds) > threshold:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
                    print("Detection:: Location -> ({}, {})".format(x, y))
                    print("Scale ->  {} | Confidence Score {} \n".format(scale, loaded_model.decision_function(fds)))
                    detections.append(
                        (int(x * (downscale ** scale)), int(y * (downscale ** scale)),
                         loaded_model.decision_function(fds),
                         int(windowSize[0] * (downscale ** scale)),  # create a list of all the predictions found
                         int(windowSize[1] * (downscale ** scale))))
                    count += 1
        scale += 0.5

    print(count)
    clone = resized.copy()
    # for (x_tl, y_tl, _, w, h) in detections:
    #     cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness=2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])  # do nms on the detected bounding boxes
    sc = [score[0] for (x, y, score, w, h) in detections]
    print("detection confidence score: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.2)

    # the peice of code above creates a raw bounding box prior to using NMS
    # the code below creates a bounding box after using nms on the detections
    # you can choose which one you want to visualise, as you deem fit... simply use the following function:
    # cv2.imshow in this right place (since python is procedural it will go through the code line by line).

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("Raw Detections after NMS", img)
    cv2.imwrite("airplane_1.bmp", img)
    #### Save the images below
    cv2.waitKey(0) & 0xFF


def sliding_window(image, stepSize,
                   windowSize):  # image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0],
                   stepSize):  # this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])


def create_window(winname, width, height):
    cv2.namedWindow(winname, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(winname, width, height)


if __name__ == '__main__':
    main()
