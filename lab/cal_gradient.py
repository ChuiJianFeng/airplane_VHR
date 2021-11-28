import numpy as np
import cv2
from matplotlib import pyplot as plt

kernel_x = np.array([  # 下一個減掉前一個
    [-1, 0, 1],
])
kernel_y = kernel_x.transpose((1, 0))


def main():
    create_window("magnitude", 200, 200)
    create_window("original", 200, 200)
    image = cv2.imread("../resources/test1124.bmp", cv2.IMREAD_GRAYSCALE)


    gx, gy = get_image_gradient(image)
    magnitude, angle = get_magnitude_angle(gx, gy)

    hist_bin = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
    hist_g = get_hist_of_grad(magnitude, angle, hist_bin)  # histogram2D


    cv2.imshow("original", image)
    cv2.imshow("magnitude", magnitude)

    cv2.waitKey()


def get_image_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.sqrt(image / np.max(image))  # gamma校正
    gx = cv2.filter2D(image, -1, kernel_x)
    gy = cv2.filter2D(image, -1, kernel_y)

    return gx, gy


def get_magnitude_angle(gx: np.ndarray, gy: np.ndarray):
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * (180 / np.pi) % 180

    return magnitude, angle


def get_hist_of_grad(magnitude, angle, hist_bins):
    hist_g = np.zeros(shape=hist_bins.size)
    row, col = magnitude.shape
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
    print(hist_g)
    print(hist_bins)
    hist_g = hist_g / sum(hist_bins)    # 歸一化
    plt.bar(hist_bins,
            hist_g,
            width=10,
            bottom=None,
            align='center',
            color=['lightsteelblue',
                   'cornflowerblue',
                   'royalblue',
                   'midnightblue',
                   'navy',
                   'darkblue',
                   'mediumblue'])
    plt.xticks(rotation='horizontal')
    plt.show()
    return hist_g


def create_window(winname, width, height):
    cv2.namedWindow(winname, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(winname, width, height)


if __name__ == '__main__':
    main()
