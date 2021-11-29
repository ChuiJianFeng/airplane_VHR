import os
import cv2


class ImageRename():
    def __init__(self):
        self.path = '../resources/set1/airplane'  # 圖片存放地址

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        i = 1

        for item in filelist:
            if item.endswith('.bmp'):
                src = os.path.join(os.path.abspath(self.path), item)
                if i < 1000:
                    dst = os.path.join(os.path.abspath(self.path), str(i) + '.bmp')
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
            i = i + 1

        print('total %d to rename & converted %d pngs' % (total_num, i))

    def main(self):
        data_dir_path = u"../resources/set1/airplane"
        file_list = os.listdir(r'../resources/set1/airplane')
        count = 0
        for file_name in file_list:
            root, ext = os.path.splitext(file_name)
            if ext == u'.bmp' :
                abs_name = data_dir_path + '/' + file_name
                image = cv2.imread(abs_name)
                # 在下面寫要做的處理 在這邊是做修改圖片大小的處理
                img = cv2.resize(image, (64, 64))
                cv2.imwrite(abs_name, img)

if __name__ == '__main__':
    newname = ImageRename()
    # newname.rename()
    newname.main()
