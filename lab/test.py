import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def Dgray():
    img_c1 = cv2.imread("../resources/test1124.bmp", cv2.IMREAD_GRAYSCALE)

    h, w = img_c1.shape
    cx, cy = [w // 2, h // 2]

    x, y = np.meshgrid(np.arange(w) - cx, np.arange(h) - cy)  # 生成一個座標矩陣

    circular_patch = img_c1.copy()
    # circular_patch[x ** 2 + y ** 2 >= (min(cx, cy)) ** 2] = 0   #畫圓
    circle = x ** 2 + y ** 2

    arr = []
    circular_patch[np.bitwise_and(circle >= 25, circle <= 36)] = 255

    for i in range(h):
        for j in range(w):
            if 36 >= ((x[i][j] ** 2) + (y[i][j] ** 2)) >= 25:
                arr.append(circular_patch[x[i][j], y[i][j]])

    f = np.array(arr)
    a = 0
    for i in range(len(arr)):
        otheta = (np.arange(len(arr)) / len(arr)) * 8 * np.pi
        a += (f * np.cos(otheta)).sum() ** 2 + (f * np.sin(otheta)).sum() ** 2
    print(round(a))

    img = np.array(a)
    print(len(img))

    f = (img - np.min(img) / (np.max(img) - np.min(img))) * 255

    # plt.subplot(151)
    # plt.imshow(circular_patch, "gray")
    # plt.title("Original Image")

    #plt.subplot(152)
    plt.plot(img)
    plt.show()
    plt.title(" Spectral")

    # plt.subplot(153)
    # plt.imshow(np.log(1 + np.abs(f)), "gray")
    # plt.title(" Spectral")
    # plt.show()


if __name__ == '__main__':
    # main()
    Dgray()
    # draw()
