import os
import re
import sys
import time

import tensorflow.python.platform
import tensorflow as tf

from tensorflow.python.platform import gfile
import numpy as np
FLAGS = tf.app.flags.FLAGS
#FROM CIFAR
def _generate_image_and_label_batch(image, label):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 3] of type.float32.
    label: 1-D Tensor of type.int32

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'FLAGS.batch_size' images + labels from the example queue.
  num_preprocess_threads = 12
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=FLAGS.batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.min_queue_size + 3 * FLAGS.batch_size,
      min_after_dequeue=FLAGS.min_queue_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)
  return images, tf.reshape(label_batch, [FLAGS.batch_size])

def get_filenames(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')

    if not eval_data:
        first_part_dir = 'train'
    else:
        first_part_dir = 'val'
    data_dir = os.path.join(FLAGS.data_dir, first_part_dir)

    home_dir = os.getcwd()
    filenames = []
    for dirName, subdirlist, filelist in os.walk(data_dir):
        for fname in filelist:
            if not os.path.isdir(fname):
                data_file_path = os.path.join(dirName, fname)
                #data_file_path = first_part_dir + os.path.join(dirName.split(data_dir)[-1], fname)
                filenames.append(data_file_path)
    os.chdir(home_dir)
    return filenames


def load_labels(eval_data):
    if not eval_data:
        filename = os.path.join(FLAGS.label_dir, 'train.txt')
    else:
        filename = os.path.join(FLAGS.label_dir, 'val.txt')
    labels = {}
    with gfile.GFile(filename, 'r') as f:
        for line in f:
            image_file,label = line.split()
            image_file = os.path.join(FLAGS.data_dir, image_file)
            labels[image_file] = int(label)
    return labels

def queue_files(filenames, label_dict):
    np.random.shuffle(filenames)
    label_list = [label_dict[f] for f in filenames]
    lv = tf.constant(label_list)

    label_fifo = tf.FIFOQueue(len(filenames),tf.int32,shapes=[[]])
    file_fifo = tf.train.string_input_producer(filenames, shuffle=False, capacity=len(filenames))
    label_enqueue = label_fifo.enqueue_many([lv])
    return file_fifo, label_enqueue, label_fifo

def read_image(file_fifo, label_fifo):
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
def crop_and_distort_image(image):

    height = FLAGS.image_size
    width = FLAGS.image_size
    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.image.random_crop(image, [height, width])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    order = np.random.randint(2)
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation
    if order == 0:
        distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    else:
        distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
        distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)
    return float_image
def crop_image(image):

    height = FLAGS.image_size
    width = FLAGS.image_size
    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(image,
                                                         width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)
    return float_image
def get_image_batch(file_fifo, label_fifo, distorted):

    min_fraction_of_examples_in_queue = 0.4

    read_input = read_image(file_fifo, label_fifo)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)


    if distorted:
        float_image = crop_and_distort_image(reshaped_image)
    else:
        float_image = crop_image(reshaped_image)

    # Ensure that the random shuffling has good mixing properties.
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label)

def inputs(eval_data, distorted=False):
    if eval_data and distorted:
        raise RuntimeError('Dont distort inputs for evaluation!')
    filenames = get_filenames(eval_data)
    label_dict = load_labels(eval_data)
    file_fifo, label_enqueue, label_fifo = queue_files(filenames, label_dict)
    input_image,label = get_image_batch(file_fifo, label_fifo, distorted)
    return label_enqueue, input_image, label
