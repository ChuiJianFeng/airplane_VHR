import os
import cv2
import argparse

import numpy as np
from sklearn.svm import LinearSVC
from skimage import feature, exposure
import joblib

# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='what folder to use for HOG description',
                    choices=['pos_image', 'neg_image'])
args = vars(parser.parse_args())

images = []
labels = []
posNum = 227
negNum = 915

# 處理正樣本
for i in range(posNum):
    filename = '../resources/train/pos_image/' + str(i + 1) + '.bmp'
    img = cv2.imread(filename)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = np.sqrt(gray_image / np.max(gray_image))  # gamma校正
    # get the HOG descriptor for the image

    fd = feature.hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

    # Rescale histogram for better display
    # update the data and labels
    images.append(fd)
    labels.append(1)

# 處理負樣本
for i in range(posNum):
    filename = '../resources/train/neg_image/' + str(i + 1) + '.bmp'
    img = cv2.imread(filename)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = np.sqrt(gray_image / np.max(gray_image))  # gamma校正
    # get the HOG descriptor for the image
    fd = feature.hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

    # update the data and labels
    images.append(fd.reshape(-1))
    labels.append(-1)

print(len(images[0]))
# train Linear SVC
print('Training on train images...')
svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(images, labels)
joblib.dump(svm_model, '../lab/airplane_model')
