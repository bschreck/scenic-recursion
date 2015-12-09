from __future__ import division
from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import load_input
import model
import os
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/local/miniplaces/images',
                           """Path to the miniplaces data directory.""")
tf.app.flags.DEFINE_string('label_dir', '/local/miniplaces/development_kit/data',
                           """Path to the miniplaces label directory.""")
tf.app.flags.DEFINE_string('train_dir', '/local/miniplaces/train_output',
                           """Path to the miniplaces data directory.""")
tf.app.flags.DEFINE_integer('image_size', 100,"""width of image to crop to for training""")

tf.app.flags.DEFINE_integer('num_classes', 100,"""Number of classes""")
#tf.app.flags.DEFINE_integer('num_epochs', 20,"""Number of time to run through data""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 100000,"""Number of examples per epoch for train""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_eval', 10000,"""Number of examples per epoch for eval""")
tf.app.flags.DEFINE_integer('min_queue_size', 100,"""Number of examples per epoch for eval""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('tower_name', 'tower', """tower name for multi-gpu version""")

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,"""The decay to use for the moving average""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 350.0,"""Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,"""Learning rate decay factor.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,"""Initial learning rate.""")

tf.app.flags.DEFINE_string('eval_dir', '/local/miniplaces/train_output',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/local/miniplaces/train_output',
                           """Directory where to read model checkpoints.""")
#tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
#                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

def eval_once(saver, summary_writer, top_k_op, summary_op, label_enqueue):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print 'No checkpoint file found'
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size

      sess.run(label_enqueue)
      step = 0
      while step < num_iter and not coord.should_stop():
        end_epoch = False
        if step > 0:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                size = qr._queue.size().eval()
                if size - FLAGS.batch_size < FLAGS.min_queue_size:
                    end_epoch = True
        if end_epoch:
            sess.run(label_enqueue)
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = float(true_count) / float(total_sample_count)
      print '%s: precision @ 1 = %.3f' % (datetime.now(), precision)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception, e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/gpu:0'):
    # Get images and labels for CIFAR-10.
    eval_data = True
    label_enqueue, images, labels = load_input.inputs(eval_data,distorted=False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay)
    variables_to_restore = {}
    for v in tf.all_variables():
      if v in tf.trainable_variables():
        restore_name = variable_averages.average_name(v)
      else:
        restore_name = v.op.name
      variables_to_restore[restore_name] = v
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op, label_enqueue)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
