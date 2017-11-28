"""
复制文件
"""
import os

# 单个文件夹复制数量
import shutil

NUM = 100


def copy_img(olddir, destdir):
    """
    复制文件
    :param olddir: olddir/label/1.jpg
    :param destdir: 目标文件夹
    :return:
    """
    for dir in os.listdir(olddir):
        # 不复制文件
        if 'txt' in dir:
            continue

        odir = os.path.join(olddir, dir)
        ndir = os.path.join(destdir, dir)
        if not os.path.exists(ndir):
            os.mkdir(ndir)
        i = 1
        for img in os.listdir(odir):
            if '(' not in img and ')' not in img:
                oimg = os.path.join(odir, img)
                nimg = os.path.join(ndir, str(i) + '.jpg')
                shutil.copy(oimg, nimg)
                print('copy :' + odir + ' ' + str(i))
                i += 1
                if i > NUM:
                    break


if __name__ == '__main__':
    olddir = 'E:/dl_data/car_all/train/'
    destdir = 'E:/dl_data/vehicle/train/'
    if not os.path.exists(destdir):
        os.mkdir(destdir)
    copy_img(olddir, destdir)
