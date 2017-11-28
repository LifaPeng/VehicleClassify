"""
将图片数据转为tfrecord格式，类似于
"""
import os
import random
import sys

from PIL import Image

import tensorflow as tf

# 图片数据文件夹
IMG_DIR = 'E:/dl_data/vehicle/train/'
# 存放tfrecord文件的位置
TF_PATH = 'E:/dl_data/vehicle/data299/'
# 存放label索引文件位置
LABEL_FILE = 'E:/dl_data/vehicle/data299/labels.txt'

# 每个rf_record 1024张左右图片
NUM_PER_RECORD = 1024
# 图片默认大小
IMAGE_SIZE = 299
# 验证集数量
VAILDATION_NUM = 100


def get_filepaths_and_labels(data_dir):
    """
    获取图片路径和labels
    :param data_dir:
    :return: [filepaths], [labels_dict: key标签,value索引]
    """
    if not os.path.exists(data_dir):
        raise ValueError('cannot find the dir: ' + data_dir)

    filepaths = []
    labels_dict = {}

    index = 0
    for labeldir in os.listdir(data_dir):
        namedir = os.path.join(data_dir, labeldir)
        if os.path.isfile(namedir):
            continue
        for file in os.listdir(namedir):
            file = os.path.join(namedir, file)

            # 小于4k 的图片可能不完整不要
            if os.path.getsize(file) / 1024 < 4:
                continue
            filepaths.append(file)
            if labeldir not in labels_dict:
                labels_dict[labeldir] = index
                index = index + 1

    return filepaths, labels_dict


def convert_data_to_record(tf_path, filepaths, labels_dict, spilt='train'):
    """
    转换数据格式
    :param tf_path: 存放tf文件的位置
    :param filepaths: 图片路径列表
    :param labels_dict: 如: car:0 bus:1 ...组成的字典
    :param spilt: 使用类型: train、val or test
    :return:
    """

    # 每个文件大概包含1024个记录
    num_record = len(filepaths) // NUM_PER_RECORD + 1
    for i in range(num_record):
        tfname = spilt + ' _%d' % i + '_.tfrecord'
        print('start convert ' + spilt + '' + str(i + 1) + ' / ' + str(num_record) + ' to tfrecord\n')

        with tf.python_io.TFRecordWriter(os.path.join(tf_path, tfname)) as writer:
            start_ndx = i * NUM_PER_RECORD
            end_ndx = min((i + 1) * NUM_PER_RECORD, len(filepaths))

            for j in range(start_ndx, end_ndx):
                img = Image.open(filepaths[j])
                # 获取目录名作为label
                img_label = os.path.split(os.path.dirname(filepaths[j]))[1]
                # 输出查验
                if not j % 100:
                    print(filepaths[j], img_label)
                img_label = labels_dict[img_label]

                # 如果不是三通道转为三通道
                if len(img.layer) < 3:
                    img = img.convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_data = img.tobytes()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_label])),
                    'heigth': tf.train.Feature(int64_list=tf.train.Int64List(value=[IMAGE_SIZE])),
                    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[IMAGE_SIZE]))
                }))
                writer.write(example.SerializeToString())


def write_label_file(labels_dict, label_file):
    """
    将label和其索引存到文件
    :param labels_dict:
    :param label_file:
    :return:
    """
    with tf.gfile.Open(label_file, 'w') as f:
        for label in labels_dict:
            num = labels_dict[label]
            f.write('%d:%s\n' % (num, label))


def read_label_file(labelfile):
    """
    :param labelfile:
    :return: 返回按dict: [index: label]
    """
    index_label_dict = {}
    with tf.gfile.Open(labelfile, 'r') as f:
        line = f.readline()
        num = 0
        while line:
            num = num + 1
            index, label = line.split(':')
            index_label_dict[index] = label.replace('\n', '')
            line = f.readline()

        return index_label_dict


def main(argv=None):
    data_dir = IMG_DIR

    # 获取路径和label字典
    filepaths, labels_dict = get_filepaths_and_labels(data_dir)

    # 分区
    random.seed(0)
    random.shuffle(filepaths)
    train_files = filepaths[VAILDATION_NUM:]
    validat_fies = filepaths[:VAILDATION_NUM]

    # 转格式
    if not os.path.exists(TF_PATH):
        os.mkdir(TF_PATH)
    convert_data_to_record(tf_path=TF_PATH, filepaths=train_files, labels_dict=labels_dict, spilt='train')
    convert_data_to_record(tf_path=TF_PATH, filepaths=validat_fies, labels_dict=labels_dict, spilt='val')

    # 将label字典写入文件
    write_label_file(labels_dict, LABEL_FILE)
    print('finsh')


"""测试"""
if __name__ == '__main__':
    print(sys.path[0])
    tf.app.run()
