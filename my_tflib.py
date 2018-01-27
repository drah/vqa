from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pdb

# weight initializations

def w(name, shape, dtype, init, regu=None):
    """ my interface for getting variable. """
    #return tf.get_variable(name, shape, dtype, init, regu)
    w = tf.get_variable(name, shape, dtype, init, regu)
    print(w.name, shape)
    return w

def xavier_w(name, shape, seed=None, regu=None):
    """ xavier without uniform.  """
    init = tf.contrib.layers.xavier_initializer(uniform=False, seed=seed)
    return w(name, shape, tf.float32, init, regu)

def xavieru_w(name, shape, seed=None, regu=None):
    """ xavier with uniform.  """
    init = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed)
    return w(name, shape, tf.float32, init, regu)

def xavier_conv_w(name, shape, seed=None, regu=None):
    """ xavier without uniform.  """
    init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False, seed=seed)
    return w(name, shape, tf.float32, init, regu)

def xavieru_conv_w(name, shape, seed=None, regu=None):
    """ xavier with uniform.  """
    init = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=seed)
    return w(name, shape, tf.float32, init, regu)

def in2_w(name, shape, seed=None, regu=None):
    """
    Delving Deep into Rectifiers.
    FAN_IN, factor=2.0, uniform=False.
    """
    init = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode="FAN_IN", uniform=False, seed=seed)
    return w(name, shape, tf.float32, init, regu)

def in1u_w(name, shape, seed=None, regu=None):
    """
    Convolutional Architecture for Fast Feature Embedding.
    FAN_IN, factor=1.0, uniform=True.
    """
    init = tf.contrib.layers.variance_scaling_initializer(
            factor=1.0, mode="FAN_IN", uniform=True, seed=seed)
    return w(name, shape, tf.float32, init, regu)

def avg1u_w(name, shape, seed=None, regu=None):
    """
    Convolutional Architecture for Fast Feature Embedding.
    FAN_IN, factor=1.0, uniform=True.
    """
    init = tf.contrib.layers.variance_scaling_initializer(
            factor=1.0, mode="FAN_IN", uniform=True, seed=seed)
    return w(name, shape, tf.float32, init, regu=None)

def avg3u_w(name, shape, seed=None, regu=None):
    """ for tanh """
    init = tf.contrib.layers.variance_scaling_initializer(
            factor=3.0, mode="FAN_AVG", uniform=True, seed=seed)
    return w(name, shape, tf.float32, init, regu=None)

def avg6u_w(name, shape, seed=None, regu=None):
    """ for sigmoid, softmax """
    init = tf.contrib.layers.variance_scaling_initializer(
            factor=6.0, mode="FAN_AVG", uniform=True, seed=seed)
    return w(name, shape, tf.float32, init, regu=None)

def tn_w(name, shape, seed=None, regu=None, scaling=0.1):
    init = tf.truncated_normal_initializer(0.0, scaling, seed)
    return w(name, shape, tf.float32, init, regu)

def zero_b(name, shape, seed=None):
    init = tf.constant_initializer(0.0)
    return w(name, shape, tf.float32, init)

def relu_b(name, shape, seed=None):
    init = tf.constant_initializer(0.1)
    return w(name, shape, tf.float32, init)

# activations
def linear(x):
    return x

def elu(x):
    return tf.nn.elu(x)

def relu(x):
    return tf.nn.relu(x)

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def tanh(x):
    return tf.nn.tanh(x)

def softmax(x, axis=-1):
    #x = x - tf.reduce_max(x, axis, True)
    return tf.nn.softmax(x, axis)

def batch_norm(x):
    return tf.layers.batch_normalization(x, training=True)
    #return tf.layers.batch_normalization(x, training=False)

def bn_linear(x):
    return batch_norm(x)

def bn_elu(x):
    return elu(batch_norm(x))

def bn_relu(x):
    return relu(batch_norm(x))

def bn_lrelu(x, leak=0.2):
    return lrelu(batch_norm(x), leak)

def bn_tanh(x):
    return tanh(batch_norm(x))

def bn_softmax(x, axis=-1):
    return softmax(batch_norm(x), axis)

def layer_norm(x, axis=[1,2]):
    scope = "layer_norm"
    with tf.variable_scope(scope):
        mu, var = tf.nn.moments(x, axis, keep_dims=True)
        inv = tf.rsqrt(var + 1e-8)
        normalized = (x - mu) * inv
    return normalized

def q_layer_norm(x, q_len):
    scope = "q_layer_norm"
    with tf.variable_scope(scope):
        normalized_list = []
        for i in xrange(x.get_shape().as_list()[0]):
            result = layer_norm(x[i,:q_len[i],:], [0])
            result = tf.concat([result, x[i,q_len[i]:,:]], 0)
            normalized_list.append(result)
        normalized = tf.stack(normalized_list, 0)
    return normalized

def ln_elu(x, q_len=None):
    if q_len is not None:
        return elu(q_layer_norm(x, q_len))
    else:
        return elu(layer_norm(x))

def ln_relu(x, q_len=None):
    if q_len is not None:
        return relu(q_layer_norm(x, q_len))
    else:
        return relu(layer_norm(x))



# fully connected layer

def dense(x, n_neuron, w_fn=xavier_w, b_fn=relu_b, act_fn=relu, ret_wb=False):
    w = w_fn("weight", [x.get_shape().as_list()[-1], n_neuron])
    if b_fn is not None:
        b = b_fn("bias", [n_neuron])
        a = act_fn(tf.matmul(x, w) + b)
        if ret_wb:
            return a,w,b
        else:
            return a
    else:
        a = act_fn(tf.matmul(x, w))
        if ret_wb:
            return a,w
        else:
            return a

def scaling_dense(x, n_neuron, scaling, b_fn=relu_b, act_fn=relu):
    w = tn_w("weight", [x.get_shape().as_list()[-1], n_neuron], scaling=scaling)
    if b_fn is not None:
        b = b_fn("bias", [n_neuron])
        a = act_fn(tf.matmul(x, w) + b)
    else:
        a = act_fn(tf.matmul(x, w))
    return a

# convolutional layer

def conv(x, k_shape, strides=[1,1,1,1], padding="SAME",
         w_fn=xavier_conv_w, b_fn=relu_b, act_fn=relu):
    w = w_fn("weight", k_shape)
    if b_fn is not None:
        b = b_fn("bias", k_shape[3])
        c = act_fn(tf.nn.conv2d(x, w, strides, padding) + b)
    else:
        c = act_fn(tf.nn.conv2d(x, w, strides, padding))
    return c

def scaling_conv(x, k_shape, scaling, strides=[1,1,1,1], padding="SAME",
         b_fn=relu_b, act_fn=relu):
    w = tn_w("weight", k_shape, scaling=scaling)
    if b_fn is not None:
        b = b_fn("bias", k_shape[3])
        c = act_fn(tf.nn.conv2d(x, w, strides, padding) + b)
    else:
        c = act_fn(tf.nn.conv2d(x, w, strides, padding))
    return c

# embed

def embed(x, vocab_size, embed_size, w_fn=tn_w, add_pad=False):
    if add_pad:
        w = w_fn("embed", [vocab_size+1, embed_size])
        pad_addition = tf.cast(tf.equal(x, -1), tf.int32) * (vocab_size+1)
        x = x + pad_addition
        e = tf.nn.embedding_lookup(w, x)
    else:
        w = w_fn("embed", [vocab_size, embed_size])
        e = tf.nn.embedding_lookup(w, x)
    return e

# cells and recurrent layers

lstm = tf.contrib.rnn.LSTMCell
lnlstm = tf.contrib.rnn.LayerNormBasicLSTMCell


def dynamic_bi_lstm(x, n_neuron, act_fn=tanh, seq_len=None):
    """ assert x is batch_major, aka [batch, time, ...] """

    cell_class = lstm
    with tf.variable_scope("fw"):
        cell_fw = cell_class(n_neuron, activation=act_fn, cell_clip=15.0)

    with tf.variable_scope("bw"):
        cell_bw = cell_class(n_neuron, activation=act_fn, cell_clip=15.0)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x, seq_len, dtype=tf.float32)

    return outputs, states

def dynamic_lstm(x, n_neuron, act_fn=tanh, seq_len=None):
    """ assert x is batch_major, aka [batch, time, ...] """

    cell_class = lstm
    with tf.variable_scope("fw"):
        cell_fw = cell_class(n_neuron, activation=act_fn, cell_clip=15.0)
        o, s = tf.nn.dynamic_rnn(
                cell_fw, x, seq_len, dtype=tf.float32)

    return o, s

def dynamic_lnlstm(x, n_neuron, act_fn=tanh, keep_prob=1.0, seq_len=None):
    """ assert x is batch_major, aka [batch, time, ...] """

    cell_class = lnlstm
    with tf.variable_scope("fw"):
        cell_fw = cell_class(n_neuron, activation=act_fn,
                             dropout_keep_prob=keep_prob, )
        o, s = tf.nn.dynamic_rnn(
                cell_fw, x, seq_len, dtype=tf.float32)

    return o, s

def dynamic_bi_lnlstm(x, n_neuron, act_fn=tanh, keep_prob=1.0, seq_len=None):
    """ assert x is batch_major, aka [batch, time, ...] """

    cell_class = lnlstm

    with tf.variable_scope("fw"):
        cell_fw = cell_class(n_neuron, activation=act_fn,
                             dropout_keep_prob=keep_prob)

    with tf.variable_scope("bw"):
        cell_bw = cell_class(n_neuron, activation=act_fn,
                             dropout_keep_prob=keep_prob)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x, seq_len, dtype=tf.float32)

    return outputs, states


