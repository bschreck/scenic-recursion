import gzip
import os
import re
import sys
import tarfile
import urllib
import time

import tensorflow.python.platform
import tensorflow as tf

from tensorflow.python.platform import gfile
import glob
from PIL import Image
import numpy as np
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/local/miniplaces/images',
                           """Path to the miniplaces data directory.""")
tf.app.flags.DEFINE_string('label_dir', '/local/miniplaces/development_kit/data',
                           """Path to the miniplaces label directory.""")
# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 64
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 128

#FROM CIFAR
def _generate_image_and_label_batch(image, label, min_queue_examples):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'FLAGS.batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=FLAGS.batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * FLAGS.batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)
  return images, tf.reshape(label_batch, [FLAGS.batch_size])

def get_filenames(eval_data):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')

  if not eval_data:
    filenames = glob.glob(os.path.join(FLAGS.data_dir, 'train', '*/*/*.jpg'))
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = glob.glob(os.path.join(FLAGS.data_dir, 'val', '*.jpg'))
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
  return filenames, num_examples_per_epoch


def load_labels(eval_data):
    if not eval_data:
        filename = os.path.join(FLAGS.label_dir, 'train.txt')
    else:
        filename = os.path.join(FLAGS.label_dir, 'val.txt')
    labels = {}
    with gfile.GFile(filename, 'r') as f:
        for line in f:
            image_file,label = line.split()
            image_file = os.path.join(FLAGS.data_dir,image_file)
            labels[image_file] = int(label)
    return labels

def queue_files(filenames, label_dict, num_examples_per_epoch):
    np.random.shuffle(filenames)
    label_list = [label_dict[f] for f in filenames]
    lv = tf.constant(label_list)

    label_fifo = tf.FIFOQueue(len(filenames),tf.int32,shapes=[[]])
    file_fifo = tf.train.string_input_producer(filenames, shuffle=False, capacity=len(filenames))
    label_enqueue = label_fifo.enqueue_many([lv])
    return file_fifo, label_enqueue, label_fifo

def read_image(file_fifo, label_fifo, min_queue_examples):
    class MiniplacesRecord(object):
        pass
    result = MiniplacesRecord()
    result.height = 128
    result.width = 128
    result.depth = 3
    reader = tf.WholeFileReader()
    result.key, value = reader.read(file_fifo)
    image = tf.image.decode_jpeg(value, channels=3)
    image.set_shape([128,128,3])
    result.uint8image = image
    result.label = label_fifo.dequeue()
    return result

def get_image_batch(file_fifo, label_fifo, num_examples_per_epoch):

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    read_input = read_image(file_fifo, label_fifo, min_queue_examples)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                             min_queue_examples)

def main():
  with tf.Graph().as_default():
    eval_data = False
    filenames, num_examples_per_epoch = get_filenames(eval_data)
    #TODO: filenames too large for this method
    filenames = filenames
    label_dict = load_labels(eval_data)
    file_fifo, label_enqueue, label_fifo = queue_files(filenames, label_dict, num_examples_per_epoch)

    input_image,label = get_image_batch(file_fifo, label_fifo, num_examples_per_epoch)
    print input_image.get_shape()

    #image_batches holds np arrays of np arrays of whitened, cropped images
    image_batches = []
    label_batches = []
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init_op)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord,sess=sess)
      #threads = tf.train.start_queue_runners(sess=sess)

      #basically training steps, each iteration loads an image
        #TODO: how to properly do enqueing and dequeing
      sess.run([label_enqueue])
      for i in xrange(100):
        image_batch, label_batch= sess.run([input_image, label])
        image_batches.append(image_batch)
        label_batches.append(label_batch)
        #print one_f.get_shape()
        #im = Image.fromarray(one_f)
        #im.show()

      coord.request_stop()
      coord.join(threads)
    print len(image_batches)
    print image_batches[0].shape
    print len(label_batches)
    print label_batches[0].shape
if __name__ == '__main__':
    main()
