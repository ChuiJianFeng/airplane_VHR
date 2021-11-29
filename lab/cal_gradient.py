import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import argparse
from skimage import feature
import joblib
from sklearn.svm import LinearSVC

kernel_x = np.array([  # 下一個減掉前一個
    [-1, 0, 1],
])
kernel_y = kernel_x.transpose((1, 0))


def main():
    # create_window("magnitude", 200, 200)
    # create_window("original", 200, 200)

    # initial
    posNum = 300
    negNum = 278
    images = []
    labels = []

    # positive
    for i in range(posNum):
        filename = '../resources/pos_image/' + str(i + 1) + '.bmp'
        image = cv2.imread(filename)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = np.sqrt(gray_image / np.max(gray_image))  # gamma校正

        h, w = gray_image.shape
        cx, cy = w // 2, h // 2
        x, y = np.meshgrid(np.arange(w) - cx, np.arange(h) - cy)  # 生成一個座標矩陣
        circular_patch = gray_image.copy()
        circular_patch[x ** 2 + y ** 2 >= (min(cx, cy)) ** 2] = 0

        gx, gy = get_image_gradient(circular_patch)
        magnitude, angle = get_magnitude_angle(gx, gy)

        # get hist
        hist_bin = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
        hist_g = get_hist_of_grad(magnitude, angle, hist_bin)  # 每張圖的角度sum

        # get hue and classifier
        hue = np.array([0, 0, 0, 0, 0, 0])
        result, hue = get_hue(hue, image)
        # print(f" result:{result} , hue: {hue} \r\n", end=" ")

        # build cell store hug and hist
        # cell_multi_feature = np.concatenate((np.array(hue),np.array(hist_g)), axis=0)
        # # 將該hog特征值存到featureArray里面
        images.append(hist_g.reshape(-1))
        labels.append(1)

    # negative
    for i in range(negNum):
        filename = '../resources/neg_image/' + str(i + 1) + '.bmp'
        image = cv2.imread(filename)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = np.sqrt(gray_image / np.max(gray_image))  # gamma校正

        h, w = gray_image.shape
        cx, cy = w // 2, h // 2
        x, y = np.meshgrid(np.arange(w) - cx, np.arange(h) - cy)  # 生成一個座標矩陣
        circular_patch = gray_image.copy()
        circular_patch[x ** 2 + y ** 2 >= (min(cx, cy)) ** 2] = 0

        gx, gy = get_image_gradient(circular_patch)
        magnitude, angle = get_magnitude_angle(gx, gy)

        # get hist
        hist_bin = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
        hist_g = get_hist_of_grad(magnitude, angle, hist_bin)  # 每張圖的角度sum

        # get hue and classifier
        hue = np.array([0, 0, 0, 0, 0, 0])
        result, hue = get_hue(hue, image)
        # print(f" result:{result} , hue: {hue} \r\n", end=" ")

        # build cell store hug and hist
        # cell_multi_feature = np.concatenate((np.array(hue),np.array(hist_g)), axis=0)
        # # 將該hog特征值存到featureArray里面
        images.append(hist_g.reshape(-1))
        labels.append(-1)

    # train Linear SVC
    print('Training on train images...')
    svm_model = LinearSVC(random_state=42, tol=1e-5)
    svm_model.fit(images, labels)
    print(svm_model.classes_)
    joblib.dump(svm_model, 'airplane_model')


def get_image_gradient(image: np.ndarray):  # -> tuple[np.ndarray, np.ndarray]:

    gx = cv2.filter2D(image, -1, kernel_x)
    gy = cv2.filter2D(image, -1, kernel_y)

    return gx, gy


def get_magnitude_angle(gx: np.ndarray, gy: np.ndarray):
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * (180 / np.pi) % 180
    return magnitude, angle


def get_hist_of_grad(magnitude, angle, hist_bins):
    eps = 1e-5
    hist_g = np.zeros(shape=hist_bins.size)
    row, col = magnitude.shape
    angle = angle + 0.00001
    # 丟失180為0這些數據，全部都會集中在0-20之間 數據會錯
    for i in range(row):
        for j in range(col):
            idx = (np.abs(hist_bins - angle[i][j])).argmin()
            if hist_bins[idx] == angle[i][j]:
                hist_g[idx] += magnitude[i][j]
            else:
                if angle[i][j] > 160:
                    hist_g[len(hist_bins) - 1] += ((180 - angle[i][j]) / 20) * magnitude[i][j]  # 平均分配
                    hist_g[0] += ((angle[i][j] - hist_bins[len(hist_bins) - 1]) / 20) * magnitude[i][j]
                else:
                    if angle[i][j] > hist_bins[idx]:
                        hist_g[idx + 1] += ((angle[i][j] - hist_bins[idx]) / 20) * magnitude[i][j]
                        hist_g[idx] += ((hist_bins[idx + 1] - angle[i][j]) / 20) * magnitude[i][j]
                    else:
                        hist_g[idx - 1] += ((hist_bins[idx] - angle[i][j]) / 20) * magnitude[i][j]
                        hist_g[idx] += ((angle[i][j] - hist_bins[idx - 1]) / 20) * magnitude[i][j]
    hist_g = hist_g / sum(hist_g)  # 歸一化

    # L2-hsy
    out = hist_g / np.sqrt(np.sum(hist_g ** 2) + eps ** 2)
    out = np.minimum(out, 0.2)
    out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)

    # plt.bar(hist_bins,
    #         hist_g,
    #         width=10,
    #         bottom=None,
    #         align='center',
    #         color=['lightsteelblue',
    #                'cornflowerblue',
    #                'royalblue',
    #                'midnightblue',
    #                'navy',
    #                'darkblue',
    #                'mediumblue'])
    # plt.xticks(range(0, 180, 20), rotation='horizontal')
    # plt.show()
    return out


def get_hue(hue, image):
    rgb = [0, 0, 0]
    for row in image:
        for r, g, b in row:
            if r != 0:
                rgb[0] = r / 255
            if g != 0:
                rgb[1] = g / 255
            if b != 0:
                rgb[2] = b / 255
            max_value = max(rgb)
            min_value = min(rgb)

            if max_value - min_value == 0:
                result = 0
            else:
                if max_value == rgb[0]:
                    result = 60 * (((rgb[1] - rgb[2]) / (max_value - min_value)) % 6)
                elif max_value == rgb[1]:
                    result = 60 * (((rgb[2] - rgb[0]) / (max_value - min_value)) + 2)
                else:
                    result = 60 * (((rgb[0] - rgb[1]) / (max_value - min_value)) + 4)
            value = result
            # classifier hue
            angle = np.array([0, 60, 120, 180, 240, 300])
            if value == 360:
                hue[0] += 1
                return hue

            idx = (np.abs(angle - value)).argmin()
            if angle[idx] == value:
                hue[idx] += 1
            else:
                if value < angle[idx]:
                    hue[idx - 1] += 1
                else:
                    hue[idx] += 1

    return result, hue


def create_window(winname, width, height):
    cv2.namedWindow(winname, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(winname, width, height)


def predict():
    loaded_model = joblib.load('airplane_model')
    count = 0
    for i in range(100):
        filename = '../resources/neg_image/' + str(i + 1) + '.bmp'
        img = cv2.imread(filename)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_image = np.sqrt(gray_image / np.max(gray_image))  # gamma校正

        gx, gy = get_image_gradient(gray_image)
        magnitude, angle = get_magnitude_angle(gx, gy)

        # get hist
        hist_bin = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
        hist_g = get_hist_of_grad(magnitude, angle, hist_bin)  # 每張圖的角度sum

        pred = loaded_model.predict(hist_g.reshape(1, -1))[0]
        if pred == 1:
            count += 1

        print(pred)
    print(count)


if __name__ == '__main__':
    #main()
    predict()
