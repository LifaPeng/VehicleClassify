import argparse
import os
import sys

import tensorflow as tf
from tensorflow.contrib.framework import assign_from_checkpoint_fn

import inception_v3

# 导入上级包，执行sh时不用添加from vehicle_type
sys.path.append('../')
import load_data

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data/train299/',
                    help='Directory where the such: data_dir/labeldir/image.')

parser.add_argument('--convert_data', type=bool, default=False,
                    help='convert data to tfrecord or not')

parser.add_argument('--train_dir', type=str, default='../train_log/inceptions_v3/',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--max_steps', type=int, default=1000,
                    help='Number of batches to run.')
parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to train device placement.')
parser.add_argument('--log_frequency', type=int, default=50,
                    help='How often to train results to the console.')

IMAGE_SIZE = 299
BATCH_SIZE = 20
NUM_CLASS = 5


def add_pathend_char(path):
    """
    在路径后面添加'/'
    :return:
    """
    if path.endswith('/'):
        return path
    else:
        path += '/'
        return path


def train(data_dir, train_dir, max_steps, log_frequency):
    with tf.Graph().as_default():
        # Set up the data loading:
        images, labels = load_data.read_tfrecord(data_dir, image_size=IMAGE_SIZE,
                                                 is_train=True, batch_size=BATCH_SIZE, num_classes=NUM_CLASS)

        # Define the model:
        predictions, end_points = inception_v3.inception_v3(inputs=images,
                                                            num_classes=NUM_CLASS,
                                                            is_training=True)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Specify the loss function:
        slim.losses.softmax_cross_entropy(predictions, labels)
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        total_loss = slim.losses.get_total_loss()
        tf.summary.scalar('losses/total_loss', total_loss)

        # Specify the optimization scheme:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        train_tensor = slim.learning.create_train_op(total_loss, optimizer)

        # 使用预训练模型
        checkpoint_path = '../train_log/inceptions_v3/inception_v3.ckpt'
        # Read data from checkpoint file
        reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        exclude_var = []
        for key in var_to_shape_map:
            if 'InceptionV3/Logits' in key or 'InceptionV3/AuxLogits' in key:
                exclude_var.append(key)
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude_var)
        init_fn = assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

        # 循环训练
        slim.learning.train(train_tensor,
                            train_dir,
                            init_fn=init_fn,
                            number_of_steps=max_steps,
                            save_summaries_secs=log_frequency,
                            save_interval_secs=100,
                            log_every_n_steps=log_frequency)


def main(argv=None):  # pylint: disable=unused-argument
    data_dir = add_pathend_char(FLAGS.data_dir)
    train_dir = add_pathend_char(FLAGS.train_dir)
    max_steps = FLAGS.max_steps
    log_frequency = FLAGS.log_frequency

    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)

    train(data_dir=data_dir, train_dir=train_dir,
          max_steps=max_steps, log_frequency=log_frequency)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()
