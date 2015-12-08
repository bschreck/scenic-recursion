from __future__ import division
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
import util
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
    return glimpse




def glimpse_network(full_image, location):
    glimpse = _extract_glimpse_from_location(full_image, location)
    #glimpse of size (batch_size, glimpse_size, glimpse_size, 3)
    # conv1
    with tf.variable_scope('glimpse/image/conv1') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[3, 3, 3, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(glimpse, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), [FLAGS.batch_size, FLAGS.glimpse_size, FLAGS.glimpse_size, 64])
        conv1 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv1)

    # conv2
    with tf.variable_scope('glimpse/image/conv2') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv2 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv2)

    # conv3
    with tf.variable_scope('glimpse/image/conv3') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[7, 7, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv3 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv3)

    # fc4
    with tf.variable_scope('glimpse/image/fc4') as scope:
        total_elts = FLAGS.batch_size*FLAGS.glimpse_size*FLAGS.glimpse_size*64
        W_fc4 = util._variable_with_weight_decay('weights', shape=[total_elts, FLAGS.lstm_size],
                                           stddev=1e-4, wd=0.0)
        b_fc4 = tf.get_variable('biases', [FLAGS.lstm_size], initializer=tf.constant_initializer(0.1))

        conv3_flat = tf.reshape(conv3, [-1, total_elts])
        fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
        #_activation_summary(fc4)

    # fc1
    with tf.variable_scope('glimpse/location/fc1') as scope:
        W_fc1 = util._variable_with_weight_decay('weights', shape=[2, FLAGS.lstm_size],
                                           stddev=1e-4, wd=0.0)
        b_fc1 = tf.get_variable('biases', [FLAGS.lstm_size], initializer=tf.constant_initializer(0.1))

        location_flat = tf.reshape(location, [-1, 2])
        fc1 = tf.nn.relu(tf.matmul(location_flat, W_fc1) + b_fc1)
        #_activation_summary(fc1)

    # output feature vector
    with tf.variable_scope('glimpse/output') as scope:
        output = tf.mul(fc1, fc4)
        #_activation_summary(output)
    return output

def context_network(low_res):
    # conv1
    with tf.variable_scope('context/conv1') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(low_res, kernel, [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv1 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv1)

    # conv2
    with tf.variable_scope('context/conv2') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv2 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv2)

    # conv3
    with tf.variable_scope('context/conv3') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[7, 7, 64, 2],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [2], initializer=tf.constant_initializer(0.1))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv3 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv3)

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
        #_activation_summary(loc)
    return orig_loc, keep_going

def classification_network(state):
    with tf.variable_scope('classification/fc1') as scope:
        W_fc1 = util._variable_with_weight_decay('weights', shape=[2*FLAGS.lstm_size, FLAGS.num_classes],
                                           stddev=1e-4, wd=0.0)
        b_fc1 = tf.get_variable('biases', [FLAGS.num_classes], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(state, W_fc1) + b_fc1)
        #_activation_summary(fc1)
    with tf.variable_scope('classification/fc1') as scope:
        softmax = tf.nn.softmax(fc1)
        #_activation_summary(softmax)
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

            lstm1_output, lstm1_state = lstm1(glimpse_network(full_image, location), lstm1_state)

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

    return classifications

