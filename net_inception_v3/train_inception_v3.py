import argparse
import os
import sys

import tensorflow as tf

import time
import inception_v3

# 导入上级包，执行sh时不用添加from ...
sys.path.append('../')
from dataset import tf_convert
from dataset import load_data

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='E:/dl_data/vehicle/data299/',
                    help='Directory where the such: data_dir/labeldir/image.')

parser.add_argument('--convert_data', type=bool, default=False,
                    help='convert data to tfrecord or not')

# /image/deepai/vehicle_train/incetion/
parser.add_argument('--train_dir', type=str, default='../train_log/inception/',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--max_steps', type=int, default=10000,
                    help='Number of batches to run.')
parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to train device placement.')
parser.add_argument('--log_frequency', type=int, default=20,
                    help='How often to train results to the console.')

IMAGE_SIZE = 299
BATCH_SIZE = 20
NUM_CLASS = 5


def train(data_dir, train_dir, max_steps, log_frequency):
    train_batch, train_label_batch = load_data.read_tfrecord(data_dir, image_size=IMAGE_SIZE,
                                                             spilt='train', batch_size=BATCH_SIZE,
                                                             num_classes=NUM_CLASS)

    val_batch, val_label_batch = load_data.read_tfrecord(data_dir, image_size=IMAGE_SIZE,
                                                         spilt='val', batch_size=BATCH_SIZE,
                                                         num_classes=NUM_CLASS)

    X = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    Y = tf.placeholder(dtype=tf.int64, shape=[BATCH_SIZE, NUM_CLASS])

    # 训练集
    logits, _ = inception_v3.inception_v3(inputs=X, num_classes=NUM_CLASS, is_training=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(logits, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('train_acc', acc)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    start_time = now_time = time.time()
    with tf.Session() as sess:
        summary_train_writer = tf.summary.FileWriter(os.path.join(train_dir, 'trainsum'), sess.graph)
        summary_val_writer = tf.summary.FileWriter(os.path.join(train_dir, 'valsum'), sess.graph)

        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(max_steps):
            if coord.should_stop():
                break
            train_images, train_labels = sess.run([train_batch, train_label_batch])
            train_loss, _, train_acc, summary_str = sess.run([loss, train_op, acc, summary_op],
                                                             feed_dict={X: train_images, Y: train_labels})

            if step % log_frequency == 0:
                # 添加训练数据到tensorboard
                summary_train_writer.add_summary(summary_str, step)

                valdat_images, valdat_labels = sess.run([val_batch, val_label_batch])
                valdat_loss, valdat_acc, summary_str = sess.run([loss, acc, summary_op],
                                                                feed_dict={X: valdat_images, Y: valdat_labels})
                # 添加验证集数据到tensorboard
                summary_val_writer.add_summary(summary_str, step)

                print('Step %d using %ds, loss %f, acc %.2f%% --- * val_loss %f, val_acc %.2f%%' % (
                    step, time.time() - now_time, train_loss, train_acc * 100.0, valdat_loss, valdat_acc * 100.0))

            if step % (max_steps / 5) == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        print('all using %d time：' % (time.time() - start_time))


def main(argv=None):  # pylint: disable=unused-argument
    data_dir = FLAGS.data_dir + '/'
    train_dir = FLAGS.train_dir + '/'
    max_steps = FLAGS.max_steps
    log_frequency = FLAGS.log_frequency

    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)

    train(data_dir=data_dir, train_dir=train_dir,
          max_steps=max_steps, log_frequency=log_frequency)


if __name__ == '__main__':
    print(sys.path[0])
    FLAGS = parser.parse_args()
    tf.app.run()
