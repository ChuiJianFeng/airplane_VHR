import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

kernel_x = np.array([  # 下一個減掉前一個
    [-1, 0, 1],
])
kernel_y = kernel_x.transpose((1, 0))

def loop_data(fp, ft):
    for i in ft:
        filename = fp + str(i)  + '.bmp'
    return filename

def main(filename):
    create_window("magnitude", 200, 200)
    create_window("original", 200, 200)


    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.sqrt(gray_image / np.max(gray_image))  # gamma校正

    gx, gy = get_image_gradient(gray_image)
    magnitude, angle = get_magnitude_angle(gx, gy)

    # get hist
    hist_bin = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
    hist_g = get_hist_of_grad(magnitude, angle, hist_bin)  # 每張圖的角度sum

    # get hue and classifier
    hue = np.array([0, 0, 0, 0, 0, 0])
    result, hue = get_hue(hue, image)
    print(f" result:{result} , hue: {hue} \r\n", end=" ")

    # build cell store hug and hist
    cell_multi_feature = np.concatenate((np.array(hue),np.array(hist_g)), axis=0)



    cv2.imshow("original", image)
    cv2.imshow("magnitude", magnitude)

    cv2.waitKey()


def get_image_gradient(image: np.ndarray):  # -> tuple[np.ndarray, np.ndarray]:

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
    #print(hist_g)
    print(hist_bins)
    hist_g = hist_g / sum(hist_bins)  # 歸一化

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
    plt.xticks(range(0, 180, 20), rotation='horizontal')
    plt.show()
    return hist_g


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



if __name__ == '__main__':
    data_dir_path = "../resources/set1/airplane"
    file_list = os.listdir(r'../resources/set1/airplane')
    file_name = loop_data(data_dir_path, file_list)
    main(file_name)

