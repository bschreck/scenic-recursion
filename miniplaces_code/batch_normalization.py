import tensorflow as tf
import util
FLAGS = tf.app.flags.FLAGS

def batch_normalized_linear_layer(state_below, scope_name, n_inputs, n_outputs, stddev, wd, eps=.0001):
  with tf.variable_scope(scope_name) as scope:
    weight = _variable_with_weight_decay(
      "weights", shape=[n_inputs, n_outputs],
      stddev=stddev, wd=wd
    )
    act = tf.matmul(state_below, weight)
    # get moments
    act_mean, act_variance = tf.nn.moments(act, [0])
    # get mean and variance variables
    mean = _variable_on_cpu('bn_mean', [n_outputs], tf.constant_initializer(0.0))
    variance = _variable_on_cpu('bn_variance', [n_outputs], tf.constant_initializer(1.0))
    # assign the moments
    assign_mean = mean.assign(act_mean)
    assign_variance = variance.assign(act_variance)

    act_bn = tf.mul((act - mean), tf.rsqrt(variance + eps), name=scope.name+"_bn")

    beta = _variable_on_cpu("beta", [n_outputs], tf.constant_initializer(0.0))
    gamma = _variable_on_cpu("gamma", [n_outputs], tf.constant_initializer(1.0))
    bn = tf.add(tf.mul(act_bn, gamma), beta)
    output = tf.nn.relu(bn, name=scope.name)
    _activation_summary(output)
  return output, mean, variance