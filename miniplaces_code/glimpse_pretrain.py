from __future__ import division
import tensorflow as tf
import load_input
import glimpse_pretrain_model as model
import time
import numpy as np
from datetime import datetime
import os,sys
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
tf.app.flags.DEFINE_string('checkpoint_dir', '/local/miniplaces/train_output',
                           """Directory where to read model checkpoints.""")
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
tf.app.flags.DEFINE_string('device', '/gpu:0', """device to use for variables""")

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,"""The decay to use for the moving average""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 350.0,"""Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,"""Learning rate decay factor.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,"""Initial learning rate.""")
tf.app.flags.DEFINE_float('rms_decay', 0.9,"""RMS decay factor.""")

def train():

    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step',[],
            initializer=tf.constant_initializer(0), trainable=False)

        eval_data = False
        label_enqueue, images, labels = load_input.inputs(eval_data, distorted=True)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(images)

        # Calculate loss.
        loss = model.loss(logits, labels)

        n = tf.zeros([1], dtype=tf.int32)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=FLAGS.log_device_placement)) as sess:
            sess.run(init)

            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))
            sess.run(label_enqueue)

            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                    graph_def=sess.graph_def)



            for step in xrange(FLAGS.max_steps):
                #print 'step:',step
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                #print 'ran'
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / float(duration)
                    sec_per_batch = float(duration)

                    sys.stdout.flush()
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                    print format_str % (datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch)
                #print 'here'
                if step % 100 == 0:
                    #print "entering summary"
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                    #print "exiting summary"

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

def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
