from __future__ import division
from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import load_input
import os
FLAGS = tf.app.flags.FLAGS
import recurrent_model as model

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
                print 'end epoch:', end_epoch
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
  with tf.Graph().as_default():
    # Get images and labels for CIFAR-10.
    eval_data = True
    label_enqueue, images, labels = load_input.inputs(eval_data,distorted=False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.rnn_model(images)

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
