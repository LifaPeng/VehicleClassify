"""
    读取数据
"""
import os
import tensorflow as tf

MIN_AFTER_QUEUE = 200
NUM_THREAD = 2


def read_tfrecord(data_dir, image_size=224, spilt='train', batch_size=20, num_classes=5):
    """
    从tfrecord中读取数据
    :param data_dir: 数据集位置
    :param image_size: 图片大小
    :param spilt: 'train','val' or 'test'
    :param batch_size: 一次读取大小
    :param num_classes: Tensor张量： images， labels
    :return:
    """

    # 根据不同的需求（spilt = train/val/test）取出不同的数据
    filename_queue = []
    for file in os.listdir(data_dir):
        if file.startswith(spilt) and file.endswith('.tfrecord'):
            filename_queue.append(os.path.join(data_dir, file))

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tf.train.string_input_producer(filename_queue))

    features = tf.parse_single_example(serialized_example, features={
        'img': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'heigth': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
    })

    # 将image数据和label取出来
    image = tf.decode_raw(features['img'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 3])
    # 在流中抛出label张量
    label = tf.cast(features['label'], tf.int64)

    # 图像处理
    distorted_image = tf.random_crop(image, [image_size, image_size, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    image = tf.reshape(image, [image_size, image_size, 3])
    # 归一化处理
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # 是（train）否(test)使用不随机批处理
    # batch_size用于一次独立计算的数据量，capacity是一个队列的容量，min_after_dequeue最后出队数
    # 就是一次读capacity作为缓存，但是只输出batch_size作为计算量
    if 'train' in spilt:
        images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                capacity=MIN_AFTER_QUEUE + 3 * batch_size,
                                                num_threads=NUM_THREAD,
                                                min_after_dequeue=MIN_AFTER_QUEUE)

    else:
        images, labels = tf.train.batch([image, label],
                                        batch_size=batch_size,
                                        capacity=50,
                                        num_threads=NUM_THREAD)

    # 将labels转为ont-hot编码
    labels = tf.one_hot(labels, num_classes, 1, 0)
    labels = tf.cast(labels, dtype=tf.int32)
    labels = tf.reshape(labels, [batch_size, num_classes])

    return images, labels


def get_images_labels(data_dir, image_size=224, spilt='train'):
    """
    启动session获取数据,用于测试
    :param data_dir: 数据集
    :param image_size:
    :param spilt: 'train','val' or 'test'
    :return: 两个Tensor: images[batch_size,w,h,d],labels[batch_size] 或者单通道的images[batch_size,w,h]
    """
    images, labels = read_tfrecord(data_dir=data_dir, image_size=image_size, spilt=spilt)
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # sess.run 运行前文img和label定义的计算
        reimages, relabels = sess.run([images, labels])  # 在会话中取出image和label

        coord.request_stop()
        coord.join(threads)

        print(reimages[0], labels[0])
        return reimages, relabels


if __name__ == '__main__':
    get_images_labels(data_dir='E:/dl_data/vehicle/data224/', image_size=224, spilt='train')
