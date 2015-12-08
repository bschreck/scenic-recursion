from __future__ import division
import tensorflow as tf
import load_input
import time
import numpy as np
from datetime import datetime
import os
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/local/miniplaces/images',
                           """Path to the miniplaces data directory.""")
tf.app.flags.DEFINE_string('label_dir', '/local/miniplaces/development_kit/data',
                           """Path to the miniplaces label directory.""")
tf.app.flags.DEFINE_string('train_dir', '/local/miniplaces/train_output',
                           """Path to the miniplaces data directory.""")
tf.app.flags.DEFINE_integer('image_size', 100,"""width of image to crop to for training""")
tf.app.flags.DEFINE_integer('glimpse_size', 32,"""width of image to extract glimpse for rnn step""")
tf.app.flags.DEFINE_integer('context_image_size', 32,"""width of downsampled image to feed into context layer""")

tf.app.flags.DEFINE_integer('lstm_size', 256,"""size of lstm state""")

tf.app.flags.DEFINE_integer('num_classes', 100,"""Number of classes""")
#tf.app.flags.DEFINE_integer('num_epochs', 20,"""Number of time to run through data""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 100000,"""Number of examples per epoch for train""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_eval', 10000,"""Number of examples per epoch for eval""")
tf.app.flags.DEFINE_integer('min_queue_size', 100,"""Number of examples per epoch for eval""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_recurrent_steps', 5,
                            """Max number of glimpses.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('tower_name', 'tower', """tower name for multi-gpu version""")
tf.app.flags.DEFINE_string('device', '/gpu:0', """device to use for variables""")

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,"""The decay to use for the moving average""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 350.0,"""Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,"""Learning rate decay factor.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,"""Initial learning rate.""")
tf.app.flags.DEFINE_float('float_to_pixel', .15, """Ratio of location unit width to number of pixels""")
tf.app.flags.DEFINE_float('keep_going_threshold', .1, """Ratio of location unit width to number of pixels""")

import recurrent_model as model

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step',[],
            initializer=tf.constant_initializer(0), trainable=False)

        eval_data = False
        label_enqueue, images, labels = load_input.inputs(eval_data, distorted=True)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.rnn_model(images)

        # # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

            # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=False,
                    log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        sess.run(label_enqueue)

        for step in xrange(32):
            start_time = time.time()
            print "running:"
            logit_outputs = sess.run(logits)

            n = tf.zeros([1], dtype=tf.int32)
            logit_outputs = tf.Print(logit_outputs, [logit_outputs], message='outputs:')
            print "ran:"
            print logit_outputs.get_shape()
            duration = time.time() - start_time
        print duration

        coord.request_stop()
        coord.join(threads)

def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
