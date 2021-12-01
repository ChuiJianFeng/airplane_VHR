# load the model from disk
import joblib
import cv2
from sklearn.svm import LinearSVC
from skimage import feature
import numpy as np

kernel_x = np.array([  # 下一個減掉前一個
    [-1, 0, 1],
])
kernel_y = kernel_x.transpose((1, 0))


def main():
    loaded_model = joblib.load('airplane_model')
    tp = 0
    fp = 0
    num = 83
    for i in range(num):
        filename = '../test/neg/' + str(i + 1) + '.bmp'
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
            tp += 1
        else:
            fp += 1

        print(f'{i + 1} : {pred} \n', end=" ")
    print(tp)


def get_image_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    return out


if __name__ == '__main__':
    main()
