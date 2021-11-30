import numpy as np
import cv2
from PIL import Image

# size = 400
# sliding = 200
size = 100
sliding = 50
sliding_merge = 100

def merge(height, weight):
    for j in range(0, height, sliding):
        for i in range(0, weight, sliding):
            name = 'number' + str(j) + '-' + str(i) + '_out.bmp'
            img1 = cv2.imread(name)
            if i == 0:
                test = img1
            elif (size + i  > weight):
                test = test[:, 0:weight - (size - sliding_merge), :]
                img1 = img1[:, sliding_merge:size, :]
                test = cv2.hconcat([test, img1])
                break;
            else:
                img1 = img1[:, size - (sliding + sliding_merge):size, :]
                test = test[:, 0:test.shape[1] - sliding_merge, :]
                test = cv2.hconcat([test, img1])

        if (j == 0):
            result = test
        elif (size + j  > height):
            result = result[0:height - (size - sliding_merge), :, :]
            test = test[sliding_merge:size, :, :]
            result = cv2.vconcat([result, test])
            break
        else:
            result = result[0:result.shape[0] - sliding_merge, :, :]
            test = test[size - (sliding + sliding_merge):size, :, :]
            result = cv2.vconcat([result, test])
    return result


def sliding_window_cut(img):
    t= False
    cnt = 100
    # cnt = 1
    for j in range(0, img.shape[0], sliding):
        for i in range(0, img.shape[1], sliding):
            # name = '0' + str(cnt) + '.bmp'
            # name = 'aircraft_90__' + str(cnt) + '.bmp'
            cnt+=1
            name = 'number' + str(j) + '-' + str(i) + '.bmp'
            if(size + j > img.shape[0]):
                if (size + i > img.shape[1]):
                    partImg = img[img.shape[0] - size: img.shape[0], img.shape[1] - size: img.shape[1], :]
                    cv2.imwrite(name, partImg)
                    t=True
                    break;
                partImg = img[img.shape[0] - size: img.shape[0], i: size + i, :]
            else:
                if (size + i > img.shape[1]):
                    partImg = img[j: size + j, img.shape[1] - size: img.shape[1], :]
                    cv2.imwrite(name, partImg)
                    break;
                partImg = img[j: size + j, i: size + i, :]

            cv2.imwrite(name, partImg)
        if(t):
            break




if __name__ == '__main__':
    img = cv2.imread("../resources/JPEGImages/aircraft_831.jpg")
    shape = img.shape
    print(shape)
    # # resize to 8 multiple
    # height, weight = shape[0], shape[1]
    # if (height % 16 != 0):
    #     height = (int(height / 16) + 1) * 16
    # if (weight % 16 != 0):
    #     weight = (int(weight / 16) + 1) * 16
    #
    # img = cv2.resize(img, [weight, height])
    # shape = img.shape

    ## cut to 512X512
    # check height and weight
    isDivide_h = True
    isDivide_w = True
    if (shape[0] % 512 == 0):
        height = int(shape[0] / 512)
    else:
        height = int(shape[0] / 512) + 1
        isDivide_h = False
    if (shape[1] % 512 == 0):
        weight = int(shape[1] / 512)
    else:
        weight = int(shape[1] / 512) + 1
        isDivide_w = False


    #sliding window cut img
    sliding_window_cut(img)
    # # merge
    # img = merge(shape[0], shape[1])
    # cv2.imwrite('211028.bmp',img)