"""
评价
"""
import sys
import math

import os
import tensorflow as tf
import alexnet

slim = tf.contrib.slim

# 导入上级包，执行sh时不用添加from ...
sys.path.append('../')
from dataset import load_data

DATA_DIR = 'E:/dl_data/vehicle/data224/'
LOG_DIR = 'E:/python/code/vehicle_classify/train_log/alexnet/'
IMAGE_SIZE = 224
BATCH_SIZE = 10
NUM_CLASS = 5


def evalalexnet():
    # Create model and obtain the predictions:
    imgs, labels = load_data.read_tfrecord(data_dir=DATA_DIR, image_size=IMAGE_SIZE,
                                           spilt='val', batch_size=BATCH_SIZE,
                                           num_classes=NUM_CLASS)

    X = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    Y = tf.placeholder(dtype=tf.int64, shape=[BATCH_SIZE, NUM_CLASS])

    logits, _ = alexnet.alexnet_v2(inputs=X, num_classes=5)
    predictions = tf.nn.softmax(logits)
    predictions = tf.cast(predictions, tf.int32)

    accuracy = slim.metrics.streaming_accuracy(predictions, Y),
    precision = slim.metrics.streaming_precision(predictions, Y),
    # recall = slim.metrics.streaming_recall_at_k(predictions, labels, 5),
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('precision', precision)

    num_examples = 100
    num_batches = math.ceil(num_examples / float(BATCH_SIZE))

    summary_op = tf.summary.merge_all()
    ckptfile = '../train_log/alexnet/model.ckpt-4900'

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'eval'), sess.graph)
        tf.train.Saver().restore(sess, ckptfile)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for batch_id in range(10):
            reaccuracy, reprecision = sess.run([accuracy, precision], feed_dict={X: imgs, Y: labels})
            # summary_writer.add_event(summary_str)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    evalalexnet()
