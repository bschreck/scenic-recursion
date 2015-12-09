import tensorflow as tf
import re
import numpy as np
FLAGS = tf.app.flags.FLAGS

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

# def _variable_with_weight_decay(name, shape, stddev, wd):
  # """Helper to create an initialized Variable with weight decay.

  # Note that the Variable is initialized with a truncated normal distribution.
  # A weight decay is added only if one is specified.

  # Args:
    # name: name of the variable
    # shape: list of ints
    # stddev: standard deviation of a truncated Gaussian
    # wd: add L2Loss weight decay multiplied by this float. If None, weight
        # decay is not added for this Variable.

  # Returns:
    # Variable Tensor
  # """
  # var = tf.get_variable(name, shape,
                         # initializer=tf.truncated_normal_initializer(stddev=stddev))
  # if wd:
    # weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    # tf.add_to_collection('losses', weight_decay)
  # return var
def _xavier_variable(name, shape, fan_in=None,fan_out=None, wd=0.0):
    if not fan_in:
        fan_in = shape[0]*shape[1]*shape[2]
    if not fan_out:
        fan_out = shape[0]*shape[1]*shape[3]
    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    var = tf.get_variable(name, shape, dtype=tf.float32, initializer = tf.random_uniform_initializer(minval=low, maxval=high))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _xavier_variable('weights', shape=[5, 5, 3, 64], fan_in=5*5*3, fan_out=5*5*64)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _xavier_variable('biases', [64], fan_in=1, fan_out=5*5*64)
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    conv1 = tf.nn.relu(bias, name=scope.name)
    dropped_conv1 = tf.nn.dropout(conv1, .8)
    _activation_summary(dropped_conv1)

  # pool1
  pool1 = tf.nn.max_pool(dropped_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    # kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         # stddev=1e-4, wd=0.0)
    kernel = _xavier_variable('weights', shape=[5, 5, 64, 64], fan_in=5*5*64, fan_out=1)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    #biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
    biases = _xavier_variable('biases', [64], fan_in=1, fan_out=5*5*64)
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    conv2 = tf.nn.relu(bias, name=scope.name)
    dropped_conv2 = tf.nn.dropout(conv2, .8)
    _activation_summary(dropped_conv2)

  # norm2
  norm2 = tf.nn.lrn(dropped_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    # kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         # stddev=1e-4, wd=0.0)
    kernel = _xavier_variable('weights', shape=[7, 7, 64, 64], fan_in=7*7*64, fan_out=1)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='VALID')
    #biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
    biases = _xavier_variable('biases', [64], fan_in=1, fan_out=7*7*64)
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    conv3 = tf.nn.relu(bias, name=scope.name)
    dropped_conv3 = tf.nn.dropout(conv3, .8)
    _activation_summary(dropped_conv3)

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

    # weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          # stddev=0.04, wd=0.004)
    weights = _xavier_variable('weights', shape=[dim,256], fan_in=dim,fan_out=1, wd=.004)
    #biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1))
    biases = _xavier_variable('biases', [256], fan_in=1, fan_out=256)
    fc4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, weights), biases), name=scope.name)
    dropped_fc4 = tf.nn.dropout(fc4, .8)
    _activation_summary(dropped_fc4)


  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    # weights = _variable_with_weight_decay('weights', [192, FLAGS.num_classes],
                                          # stddev=1/192.0, wd=0.0)
    weights = _xavier_variable('weights', shape=[256,FLAGS.num_classes], fan_in=256,fan_out=1)
    # biases = tf.get_variable('biases', [FLAGS.num_classes],
                              # initializer=tf.constant_initializer(0.0))
    biases = _xavier_variable('biases', [FLAGS.num_classes], fan_in=1, fan_out=FLAGS.num_classes)
    softmax_linear = tf.nn.xw_plus_b(dropped_fc4, weights, biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


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
  # lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                  # global_step,
                                  # decay_steps,
                                  # FLAGS.learning_rate_decay_factor,
                                  # staircase=True)
  # tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.RMSPropOptimizer(FLAGS.initial_learning_rate, FLAGS.rms_decay, momentum=0.0, epsilon=1e-6, use_locking=False, name='RMSProp')
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



