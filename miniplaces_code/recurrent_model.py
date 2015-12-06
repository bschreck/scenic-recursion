import tensorflow as tf
import util
FLAGS = tf.app.flags.FLAGS

#RNN SKELETON CODE

def glimpse_network(full_image, location):
    #kernel and shapes all screwed up obviously
    glimpse = _extract_glimpse_from_location(full_image, location)
    # conv1
    with tf.variable_scope('glimpse/image/conv1') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(glimpse, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    # conv2
    with tf.variable_scope('glimpse/image/conv2') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # conv3
    with tf.variable_scope('glimpse/image/conv3') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)

    # fc4
    with tf.variable_scope('glimpse/image/fc4') as scope:
        W_fc4 = weight_variable([7 * 7 * 64, 1024])

        W_fc4 = util._variable_with_weight_decay('weights', shape=[7*7*64, 1024],
                                           stddev=1e-4, wd=0.0)
        b_fc4 = tf.get_variable('biases', [1024], initializer=tf.constant_initializer(0.1))

        conv3_flat = tf.reshape(conv3, [-1, 7*7*64])
        fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
        _activation_summary(fc4)

    # fc1
    with tf.variable_scope('glimpse/location/fc1') as scope:
        W_fc1 = weight_variable([7 * 7 * 64, 1024])

        W_fc1 = util._variable_with_weight_decay('weights', shape=[7*7*64, 1024],
                                           stddev=1e-4, wd=0.0)
        b_fc1 = tf.get_variable('biases', [1024], initializer=tf.constant_initializer(0.1))

        conv3_flat = tf.reshape(location, [-1, 7*7*64])
        fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)
        _activation_summary(fc1)

    # output feature vector
    with tf.variable_scope('glimpse/output') as scope:
        output = tf.mul(fc1, fc4)
        _activation_summary(output)
    return output

def context_network(low_res):
    # conv1
    with tf.variable_scope('context/conv1') as scope:
        kernel = util._variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(low_res, kernel, [1, 1, 1, 1], padding='SAME')
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
        kernel = util._variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)
    return conv3

def emission_network(state):
    #outputs (x,y,stop)
    #(x,y) is location tuple
    #stop is whether or not to stop recurring
    with tf.variable_scope('emission/fc1') as scope:
        W_fc1 = util._variable_with_weight_decay('weights', shape=[FLAGS.lstm_units, 3],
                                           stddev=1e-4, wd=0.0)
        b_fc1 = tf.get_variable('biases', [2], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(state, W_fc1) + b_fc1)
        _activation_summary(fc1)
    return fc1

def classification_network(state):
    with tf.variable_scope('classification/fc1') as scope:
        W_fc1 = util._variable_with_weight_decay('weights', shape=[FLAGS.lstm_units, FLAGS.num_classes],
                                           stddev=1e-4, wd=0.0)
        b_fc1 = tf.get_variable('biases', [FLAGS.num_classes], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(state, W_fc1) + b_fc1)
        _activation_summary(fc1)
    with tf.variable_scope('classification/fc1') as scope:
        softmax = tf.nn.softmax(fc1)
        _activation_summary(softmax)
    return softmax

def rnn_model(full_image):
    with tf.variable_scope('main_recurrence') as scope:
        classifications = []
        #provide 0 initialization to lstm1
        lstm1 = rnn.LSTM(FLAGS.size)

        low_res = _downsample(full_image)
        context = context_network(low_res)
        #provide context initialization to lstm2
        lstm2 = rnn.LSTM(FLAGS.size)
        for step in xrange(FLAGS.max_steps):
            tf.get_variable_context().reuse_variables()
            emission = emission_network(lstm2.output())
            location, keep_going = _parse_emission_output(emission)

            glimpse_features = glimpse_network(full_image, location)
            lstm1.update(glimpse_features)

            if step > 0:
                with tf.variable_scope('classification%d'%step):
                    classifications.append(classification_network(lstm1.output()))
            if keep_going:
                lstm2.update(lstm1.output())
            else:
                break
        return classifications

