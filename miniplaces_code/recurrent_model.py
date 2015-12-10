from __future__ import division
import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn_cell
import util
import re
FLAGS = tf.app.flags.FLAGS

#max of either on pixel scale is floor(image_size/2) - floor(glimpse_size/2) - 1
MAX_PIXEL = FLAGS.image_size//2 - FLAGS.glimpse_size//2 - 1
MAX_LOC = FLAGS.float_to_pixel*FLAGS.image_size*MAX_PIXEL


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _xavier_variable(name, shape, fan_in=None,fan_out=None, wd=0.0):
    if not fan_in:
        fan_in = shape[0]*shape[1]*shape[2]
    if not fan_out:
        fan_out = shape[0]*shape[1]*shape[3]
    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    var = tf.get_variable(name, shape, dtype=tf.float32, initializer = tf.random_uniform_initializer(minval=low, maxval=high))
    if wd:
        l2_loss = tf.nn.l2_loss(var)
        weight_decay = tf.mul(l2_loss, wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

@tf.RegisterShape("ExtractGlimpse")
def _extract_glimpse_shape(op):
  """Shape function for the "ExtractGlimpse" op.

  All inputs and the output are matrices of the same size.
  """
  first = op.inputs[0].get_shape().with_rank(4).as_list()
  second = op.inputs[1].get_shape().with_rank(1)
  input_shape_list = []
  begin = tf.TensorShape(first[0])
  middle = tf.TensorShape(None)
  end = tf.TensorShape(first[3])
  output_shape = begin.concatenate(middle).concatenate(end)
  return [output_shape]

#RNN SKELETON CODE
def _extract_glimpse_from_location(full_image, location):
    #location is 2D tensor of size (batch_size, 2)
    #full_image is 4D tensor of size (batch_size, image_size, image_size, 3)
    #returns glimpse, a 4D tensor of size (batch_size, glimpse_size, glimpse_size, 3)
    #(x,y) of location is on Cartesian grid centered at center of full_image (floor(image_size/2), floor(image_size/2))
    #x,y is float that needs to map to pixels
    #max of either on pixel scale is floor(image_size/2) - floor(glimpse_size/2) - 1
    dividend = tf.constant(int(FLAGS.float_to_pixel*FLAGS.image_size), dtype=tf.int32, shape=[FLAGS.batch_size,2])
    pixels_from_center = tf.to_float(tf.div(tf.to_int32(location),dividend))
    #pixel_location = FLAGS.image_size//2 + pixels_from_center
    #pixel_begin = pixel_location - FLAGS.glimpse_size//2
    #glimpse = tf.slice(full_image, pixel_begin, [-1, FLAGS.glimpse_size, FLAGS.glimpse_size, -1])
    glimpse_size = tf.constant(FLAGS.glimpse_size, dtype=tf.int32, shape=[2])
    glimpse = tf.image.extract_glimpse(full_image,glimpse_size,
                            pixels_from_center, centered=True, normalized=False)
    return tf.stop_gradient(glimpse)


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def glimpse_network(full_image, location):
    glimpse = _extract_glimpse_from_location(full_image, location)
    glimpse_vars = {}
    #glimpse of size (batch_size, glimpse_size, glimpse_size, 3)

    epsilon = 0.001

    # conv1
    with tf.variable_scope('glimpse/image') as outer_scope:
        with tf.variable_scope('conv1') as scope:
            kernel1 = _xavier_variable('weights', shape=[5, 5, 3, 64], fan_in=5*5*3, fan_out=5*5*64)
            conv = tf.nn.conv2d(glimpse, kernel1, [1, 1, 1, 1], padding='SAME')
            biases1 = _xavier_variable('biases', [64], fan_in=1, fan_out=5*5*64)
            bias = tf.reshape(tf.nn.bias_add(conv, biases1), [FLAGS.batch_size, FLAGS.glimpse_size, FLAGS.glimpse_size, 64])
            conv1 = tf.nn.relu(bias, name=scope.name)
            dropped_conv1 = tf.nn.dropout(conv1, .8)

            mean1, variance1 = tf.nn.moments(conv1, [0])
            beta1 = tf.Variable(tf.constant(0.0, shape=[64]))
            gamma1 = tf.Variable(tf.constant(1.0, shape=[64]))
            bn1 = tf.mul((conv1 - mean1), tf.sqrt(variance1 + epsilon))
            bn1 = tf.add(tf.mul(bn1, gamma1), beta1)

            # _activation_summary(dropped_conv1)
            _activation_summary(bn1)
            glimpse_vars['conv1/weights:0'] = kernel1
            glimpse_vars['conv1/biases:0'] = biases1

        # pool1
        pool1 = tf.nn.max_pool(dropped_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                            name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel2 = _xavier_variable('weights', shape=[5, 5, 64, 64], fan_in=5*5*64, fan_out=1)
            conv = tf.nn.conv2d(norm1, kernel2, [1, 1, 1, 1], padding='SAME')
            biases2 = _xavier_variable('biases', [64], fan_in=1, fan_out=5*5*64)
            bias = tf.reshape(tf.nn.bias_add(conv, biases2), conv.get_shape().as_list())
            conv2 = tf.nn.relu(bias, name=scope.name)
            dropped_conv2 = tf.nn.dropout(conv2, .8)
            _activation_summary(dropped_conv2)
            glimpse_vars['conv2/weights:0'] = kernel2
            glimpse_vars['conv2/biases:0'] = biases2

        # norm2
        norm2 = tf.nn.lrn(dropped_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3
        with tf.variable_scope('conv3') as scope:
            kernel3 = _xavier_variable('weights', shape=[7, 7, 64, 64], fan_in=7*7*64, fan_out=1)
            conv = tf.nn.conv2d(pool2, kernel3, [1, 1, 1, 1], padding='VALID')
            biases3 = _xavier_variable('biases', [64], fan_in=1, fan_out=7*7*64)
            bias = tf.reshape(tf.nn.bias_add(conv, biases3), conv.get_shape().as_list())
            conv3 = tf.nn.relu(bias, name=scope.name)
            dropped_conv3 = tf.nn.dropout(conv3, .8)
            _activation_summary(dropped_conv3)
            glimpse_vars['conv3/weights:0'] = kernel3
            glimpse_vars['conv3/biases:0'] = biases3

        # norm3
        norm3 = tf.nn.lrn(dropped_conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm3')
        # pool3
        pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # fc4
        with tf.variable_scope('fc4') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            dim = 1
            for d in pool3.get_shape()[1:].as_list():
              dim *= d
            reshape = tf.reshape(pool3, [FLAGS.batch_size, dim])

            weights4 = _xavier_variable('weights', shape=[dim,FLAGS.lstm_size], fan_in=dim,fan_out=1, wd=.004)
            biases4 = _xavier_variable('biases', [FLAGS.lstm_size], fan_in=1, fan_out=FLAGS.lstm_size)
            fc4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, weights4), biases4), name=scope.name)
            dropped_fc4 = tf.nn.dropout(fc4, .8)
            _activation_summary(dropped_fc4)


    # fc1
    with tf.variable_scope('glimpse/location/fc1') as scope:
        W_fc1 = util._variable_with_weight_decay('weights', shape=[2, FLAGS.lstm_size],
                                           stddev=1e-4, wd=0.0)
        b_fc1 = tf.get_variable('biases', [FLAGS.lstm_size], initializer=tf.constant_initializer(0.1))

        location_flat = tf.reshape(location, [-1, 2])
        fc1 = tf.nn.relu(tf.matmul(location_flat, W_fc1) + b_fc1)
        dropped_fc1 = tf.nn.dropout(fc1, .8)
        _activation_summary(dropped_fc1)

    # output feature vector
    with tf.variable_scope('glimpse/output') as scope:
        output = tf.mul(dropped_fc1, dropped_fc4)
        _activation_summary(output)
    return output, glimpse_vars

def context_network(low_res):
    # conv1
    with tf.variable_scope('context/conv1') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(low_res, kernel, [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    # conv2
    with tf.variable_scope('context/conv2') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # conv3
    with tf.variable_scope('context/conv3') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[7, 7, 64, 2],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [2], initializer=tf.constant_initializer(0.1))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)

    #convert to 1-d for inputting into LSTM
    return tf.reshape(conv3, [FLAGS.batch_size, -1])

def emission_network(state):
    #outputs (x,y,stop)
    #(x,y) is location tuple
    #stop is whether or not to stop recurring
    with tf.variable_scope('emission/fc1') as scope:
        W_fc1 = util._variable_with_weight_decay('weights', shape=[FLAGS.lstm_size, 3],
                                           stddev=1e-4, wd=0.0)
        b_fc1 = tf.get_variable('biases', [3], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(state, W_fc1) + b_fc1)
    return fc1


def _parse_emission_output(emission):
    with tf.variable_scope('emission/parse') as scope:
        orig_loc = tf.slice(emission, [0,0], [-1, 2])
        keep_going = tf.slice(emission, [0,2],[-1,1])
        #rescale so fits as a proper location
        max_loc = tf.constant(MAX_LOC, dtype=tf.float32, shape=[FLAGS.batch_size,2])
        loc =  tf.mul(max_loc, tf.nn.tanh(orig_loc))   #0 -> MAX_LOC
        _activation_summary(loc)
    return orig_loc, keep_going

def classification_network(state):
    with tf.variable_scope('classification/fc1') as scope:
        W_fc1 = util._variable_with_weight_decay('weights', shape=[2*FLAGS.lstm_size, FLAGS.num_classes],
                                           stddev=1e-4, wd=0.0)
        b_fc1 = tf.get_variable('biases', [FLAGS.num_classes], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(state, W_fc1) + b_fc1)
        _activation_summary(fc1)
    with tf.variable_scope('classification/fc1') as scope:
        softmax = tf.nn.softmax(fc1)
        _activation_summary(softmax)
    return softmax

def attention(classifications, valid_classifications):
    with tf.variable_scope('attention') as scope:
        attention_weights = tf.get_variable('attention_weights', [FLAGS.batch_size, FLAGS.max_recurrent_steps], initializer=tf.constant_initializer(1/FLAGS.max_recurrent_steps))
        return tf.batch_matmul(tf.transpose(classifications, perm=[1,2,0]), tf.expand_dims(tf.mul(attention_weights, tf.to_float(tf.transpose(valid_classifications))),-1), adj_y=False)

def rnn_model(full_image):
    with tf.variable_scope('main_recurrence') as scope:
        low_res = tf.image.resize_images(full_image, FLAGS.context_image_size, FLAGS.context_image_size)
        context = context_network(low_res)

        classifications_list = []
        #provide 0 initialization to lstm1
        lstm1 = rnn_cell.BasicLSTMCell(FLAGS.lstm_size)
        lstm1_state = tf.zeros([FLAGS.batch_size, lstm1.state_size])
        lstm1_outputs = []
        lstm1_states = []

        with tf.variable_scope('lstm2') as scope:
            #provide context initialization to lstm2
            lstm2 = rnn_cell.BasicLSTMCell(FLAGS.lstm_size)
            lstm2_initial_input = tf.zeros([FLAGS.batch_size, FLAGS.lstm_size])
            lstm2_output, lstm2_state = lstm2(lstm2_initial_input, context)
            emission = emission_network(lstm2_output)
            location, keep_going = _parse_emission_output(emission)
            scope.reuse_variables()


        valid_classification_list = []

        for step in xrange(FLAGS.max_recurrent_steps):
            if step > 0:
                tf.get_variable_scope().reuse_variables()
            keep_going_threshold = tf.constant(FLAGS.keep_going_threshold, dtype=tf.float32)

            glimpse_out, glimpse_vars = glimpse_network(full_image, location)
            lstm1_output, lstm1_state = lstm1(glimpse_out, lstm1_state)

            classifications_list.append(classification_network(lstm1_state))

            valids = tf.squeeze(tf.greater(keep_going, keep_going_threshold))
            valid_classification_list.append(tf.to_int32(valids))
            if not tf.reduce_any(valids):
                break

            with tf.variable_scope('lstm2') as scope:
                scope.reuse_variables()
                lstm2_output, lstm2_state = lstm2(lstm1_output, lstm2_state)

            location, keep_going = _parse_emission_output(emission_network(lstm2_output))

    valid_classifications = tf.pad(tf.pack(valid_classification_list), tf.convert_to_tensor([[0,FLAGS.max_recurrent_steps-step-1],[0,0]]))
    classifications = tf.pad(tf.pack(classifications_list), tf.convert_to_tensor([[0,FLAGS.max_recurrent_steps-step-1],[0,0],[0,0]]))

    classifications = attention(classifications, valid_classifications)
    classifications.get_shape()

    return tf.squeeze(classifications), glimpse_vars

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Reshape the labels into a dense Tensor of
  # shape [batch_size, FLAGS.num_classes].
  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
  indices = tf.reshape(tf.range(0, FLAGS.batch_size, 1), [FLAGS.batch_size, 1])
  concated = tf.concat(1, [indices, sparse_labels])
  dense_labels = tf.sparse_to_dense(concated,
                                    [FLAGS.batch_size, FLAGS.num_classes],
                                    1.0, 0.0)

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits, dense_labels, name='cross_entropy_per_example')

  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = FLAGS.num_examples_per_epoch_for_train / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                  global_step,
                                  decay_steps,
                                  FLAGS.learning_rate_decay_factor,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.RMSPropOptimizer(lr, .9, momentum=0.0, epsilon=1e-6, use_locking=False, name='RMSProp')
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      FLAGS.moving_average_decay, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op



