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
tf.app.flags.DEFINE_string('train_dir', '/local/miniplaces/rnn_train_output',
                           """Path to the miniplaces data directory.""")
tf.app.flags.DEFINE_integer('image_size', 100,"""width of image to crop to for training""")
tf.app.flags.DEFINE_integer('glimpse_size', 64,"""width of image to extract glimpse for rnn step""")
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

tf.app.flags.DEFINE_string('eval_dir', '/local/miniplaces/rnn_train_output',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/local/miniplaces/rnn_train_output',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('pretrained_checkpoint_path', '/local/miniplaces/train_output/model.ckpt-3000',
                           """Directory where to read checkpoint for pretrained cnn network.""")
#tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
#                            """How often to run the eval.""")

tf.app.flags.DEFINE_integer('eval_interval_secs', 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('evaluate', False,
                         """Whether to evaluate or train.""")

import recurrent_model as model
import rnn_eval as eval_func

def train():
    with tf.Graph().as_default(), tf.device('/gpu:0'):
        global_step = tf.get_variable(
            'global_step',[],
            initializer=tf.constant_initializer(0), trainable=False)

        eval_data = False
        label_enqueue, images, labels = load_input.inputs(eval_data, distorted=True)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits,glimpse_vars= model.rnn_model(images)
        # Calculate loss.
        loss = model.loss(logits, labels)

        n = tf.zeros([1], dtype=tf.int32)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        pretrained_glimpse_vars = {
            u'conv1/weights': glimpse_vars['conv1/weights:0'],
            u'conv1/biases': glimpse_vars['conv1/biases:0'],
            u'conv2/weights': glimpse_vars['conv2/weights:0'],
            u'conv2/biases': glimpse_vars['conv2/biases:0'],
            u'conv3/weights': glimpse_vars['conv3/weights:0'],
            u'conv3/biases': glimpse_vars['conv3/biases:0'],
            }
        # pretrained_context_vars = {
            # u'conv1/weights:': context_vars['conv1/weights:0'],
            # u'conv1/biases:':  context_vars['conv1/biases:0'],
            # u'conv2/weights:': context_vars['conv2/weights:0'],
            # u'conv2/biases:':  context_vars['conv2/biases:0'],
            # u'conv3/weights:': context_vars['conv3/weights:0'],
            # u'conv3/biases:':  context_vars['conv3/biases:0'],
        # }
        # print "="*50
        # for var in tf.all_variables():
            # print var.name, ":", var
        pretrained_glimpse_saver = tf.train.Saver(pretrained_glimpse_vars)
        #pretrained_context_saver = tf.train.Saver(pretrained_context_vars)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

            # Start running operations on the Graph.
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=FLAGS.log_device_placement)) as sess:
            sess.run(init)

            pretrained_ckpt = FLAGS.pretrained_checkpoint_path
            pretrained_glimpse_saver.restore(sess, pretrained_ckpt)
            #pretrained_context_saver.restore(sess, pretrained_ckpt)

            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))
            sess.run(label_enqueue)

            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                    graph_def=sess.graph_def)


            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / float(duration)
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                    print format_str % (datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch)
                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

                end_epoch = False
                if step > 0:
                    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                        size = qr._queue.size().eval()
                        if size - FLAGS.batch_size < FLAGS.min_queue_size:
                            end_epoch = True
                if end_epoch:
                    sess.run(label_enqueue)
            coord.request_stop()
            coord.join(threads)

def main(_):
    if FLAGS.evaluate:
        eval_func.evaluate()
    else:
        train()

if __name__ == '__main__':
    tf.app.run()
