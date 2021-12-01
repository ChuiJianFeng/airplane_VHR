import numpy as np
import os
import glob
import cv2
import argparse as ap


def main():
    r = 8  # radius == 8
    create_window("image", 200, 200)
    image = cv2.imread("../resources/train/pos_image/1.bmp", cv2.IMREAD_GRAYSCALE)

    h, w = image.shape
    cx, cy = w // 2, h // 2

    x, y = np.abs(np.meshgrid(np.arange(w) - cx, np.arange(h) - cy))
    circleimage = image.copy()
    circleimage[np.sqrt((x - cx) ** 2 + (y - cy) ** 2) == 8] = 0

    # rc.append()
    # rc.append([x, y])

    cv2.imshow("image", circleimage)
    cv2.waitKey()


def create_window(winname, width, height):
    cv2.namedWindow(winname, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(winname, width, height)


if __name__ == '__main__':
    main()
