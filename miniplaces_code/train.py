from __future__ import division
import tensorflow as tf
import load_input
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/local/miniplaces/images',
                           """Path to the miniplaces data directory.""")
tf.app.flags.DEFINE_string('label_dir', '/local/miniplaces/development_kit/data',
                           """Path to the miniplaces label directory.""")
tf.app.flags.DEFINE_integer('image_size', 100,"""width of image to crop to for training""")

tf.app.flags.DEFINE_integer('num_classes', 100,"""Number of classes""")
tf.app.flags.DEFINE_integer('num_epochs', 20,"""Number of time to run through data""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 87000,"""Number of examples per epoch for train""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_eval', 10000,"""Number of examples per epoch for eval""")


def main():
    with tf.Graph().as_default():
        eval_data = True
        label_enqueue, input_image, label, num_examples_per_epoch = load_input.inputs(eval_data)
        num_batches = num_examples_per_epoch // FLAGS.batch_size
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord,sess=sess)
            sess.run([label_enqueue])
            for i in xrange(FLAGS.num_epochs):
                for j in xrange(num_batches):
                    image_batch, label_batch= sess.run([input_image, label])
                    print image_batch.shape
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    main()
