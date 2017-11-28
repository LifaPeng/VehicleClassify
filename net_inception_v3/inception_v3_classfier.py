"""
    Created by Peng on 2017/11/3.
"""
import tensorflow as tf
import numpy as np
import PIL.Image as Image
# from skimage import io, transform
from google.protobuf import text_format
import sys
import argparse

import inception_v3

# 导入上级包，执行sh时不用添加from ...
sys.path.append('../')
from dataset import tf_convert

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--imgfile', type=str, default='../image/car1.jpg',
                    help='imgfile to classfier.')

parser.add_argument('--train_dir', type=str, default='../train_log/inception/',
                    help='label dict file.')

parser.add_argument('--labelfile', type=str, default='E:/dl_data/vehicle/data299/labels.txt',
                    help='label dict file.')


def get_img_tensor(imgfile):
    img = Image.open(imgfile)
    if len(img.layer) < 3:
        img = img.convert('RGB')
    img = img.resize([299, 299])
    img_arry = np.array(img)

    ######  不要忘记归一化！！！！！
    img_data = tf.cast(img_arry, tf.float32) * (1. / 255) - 0.5
    ######

    img_data = tf.image.per_image_standardization(img_data)
    img_data = tf.reshape(img_data, [1, 299, 299, 3])
    return img_data


def getclasses(imgfile, train_dir, labelfile):
    with tf.Graph().as_default():
        img = get_img_tensor(imgfile)
        logits, _ = inception_v3.inception_v3(inputs=img,
                                              num_classes=5)
        predictions = tf.nn.softmax(logits)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            # ckptfile = tf.train.latest_checkpoint(train_dir)
            ckptfile = '../train_log/inception/model.ckpt-2000'
            saver.restore(sess, ckptfile)
            # 获取预测的label
            predict = sess.run(predictions)
            predict_label = predict[0]

            # 获取label顺序
            index_label_dict = tf_convert.read_label_file(labelfile)
            label_out = {}
            for i, _ in enumerate(index_label_dict):
                label = index_label_dict[str(i)]
                label_out[label] = predict_label[i]
            label_out = sorted(label_out.items(), key=lambda d: d[1], reverse=True)
            print(label_out)

            return label_out


def main(argv=None):
    imgfile = FLAGS.imgfile
    train_dir = FLAGS.train_dir
    labelfile = FLAGS.labelfile

    # 获取分类
    getclasses(imgfile=imgfile, train_dir=train_dir, labelfile=labelfile)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()
