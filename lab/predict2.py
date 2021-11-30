# load the model from disk
import numpy as np
import joblib
import cv2
from sklearn.svm import LinearSVC
from skimage import feature
count = 0
for i in range(100):
    filename = '../resources/neg_image/' + str(i + 1) + '.bmp'
    loaded_model = joblib.load('airplane_model')
    img = cv2.imread('1126.bmp')
    resized_image = cv2.resize(img, (100, 100))
    (hog_desc, hog_image) = feature.hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)
    pred = loaded_model.predict(hog_desc.reshape(1, -1))[0]
    print(pred)
    if pred == 1:
        count += 1

print(pred)
print(count)


