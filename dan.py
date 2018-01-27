from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import sys
import argparse
import pdb
import glob
import time
import datetime

import vqa_processor
import my_tflib as lib
import tfrecord_reader

FLAGS=None

""" base setting """
sample_pairs = 10000
train_pairs = 221879
all_pairs = 200394 + 23954 + 95144 + 13162

# vocab_size
q_vocab_size = 16670 # including <PAD>
a_vocab_size = 2001 # including <UNK>

#g_dropout_ratio = 0.8
g_channel = 600
g_linear = lib.linear
g_act = lib.tanh
#g_q_ln_act = lib.ln_tanh

g_bn = lib.batch_norm
g_bn_linear = lib.bn_linear
g_bn_act = lib.bn_tanh
#g_ln_act = lib.ln_tanh
#nce_samples = 128
warm_start_learning_rate = 1e-1
base_learning_rate = 1e-1

# regularizations
#eta = 0.01 # gradient_noise_eta
#v_noise = 0.001 # None if not use
#q_noise = 0.001 # None if not use
eta = None # gradient_noise_eta
v_noise = None # None if not use
q_noise = None # None if not use

train_embed = True
regu_lambda = 1e-4


def tag_name(x, post_str=""):
    return x.name[4:].replace(":", "_")+post_str

class Visual:
    def __init__(self, var_scope_str, batch_size):
        self.var_scope_str = var_scope_str
        self.batch_size = batch_size

    def __call__(self, x, keep_prob):
        with tf.variable_scope(self.var_scope_str):
            net = x
            if v_noise is not None:
                net = net + tf.random_normal(net.get_shape(), stddev=v_noise)
                print("add visual features noise, std %f" % v_noise)
        return net

    def summarize_tensor(self):
        return [tf.summary.histogram(tag_name(w), w) for w in self.weights]

    @property
    def weights(self):
        return [w for w in tf.trainable_variables()
                if self.var_scope_str in w.name]


class Embed:
    def __init__(self, var_scope_str, vocab_size, embed_size, batch_size):
        self.var_scope_str = var_scope_str
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size

    def __call__(self, x):
        with tf.variable_scope(self.var_scope_str):
            # <PAD> already in q_vocab
            # this operation is for change -1 to 16669
            x = tf.cast(tf.equal(x, -1), tf.int32) * self.vocab_size + x
            e = lib.embed(x, self.vocab_size, self.embed_size, w_fn=lib.tn_w, add_pad=False)
            if q_noise is not None:
                e = e + tf.random_normal(e.get_shape(), stddev=q_noise)
                print("add embeddings noise, std %f" % q_noise)

        self.e = e
        return e

    def summarize_embed(self):
        return [tf.summary.histogram(tag_name(w), w) for w in self.weights]

    @property
    def weights(self):
        return [w for w in tf.trainable_variables()
                if self.var_scope_str in w.name]

    @property
    def embeddings(self):
        return self.weights[0]

class Question:
    def __init__(self, var_scope_str, batch_size):
        self.var_scope_str = var_scope_str
        self.batch_size = batch_size

    def __call__(self, x, keep_prob, q_len=None):
        self.keep_prob = keep_prob
        self.q_len = q_len
        with tf.variable_scope(self.var_scope_str):
            net = x # for convenience

            outputs, states = lib.dynamic_bi_lstm(net, g_channel, g_act, seq_len=q_len)

            net = outputs[0] + outputs[1]

            net = tf.expand_dims(net, 2) # for conv

            """
            self.block_1_name = "res_block_1"
            net = self._res_block(net, g_channel, self.block_1_name)

            self.block_2_name = "res_block_2"
            net = self._res_block(net, g_channel, self.block_2_name)

            self.block_3_name = "res_block_3"
            net = self._res_block(net, g_channel, self.block_3_name)

            net = g_ln_act(net)
            """

        return net

    def _res_block(self, x, channel, scope_str):
        with tf.variable_scope(scope_str):

            with tf.variable_scope("shortcut"):
                in_channel = x.get_shape().as_list()[-1]
                if channel != in_channel:
                    shortcut = lib.conv(x, [1,1,in_channel,channel],
                                        w_fn=lib.xavieru_conv_w, b_fn=lib.zero_b,
                                        act_fn=g_linear)
                else:
                    shortcut = x

            net = x
            net = g_ln_act(net, self.q_len)
            with tf.variable_scope("unigram"):
                uni = net
                with tf.variable_scope("conv1x1"):
                    uni = lib.conv(uni, [1, 1, uni.get_shape().as_list()[-1], channel//2],
                                   w_fn=lib.xavieru_conv_w, b_fn=lib.zero_b, act_fn=g_linear)
                    uni = g_ln_act(uni, self.q_len)

                with tf.variable_scope("conv1x1_2"):
                    uni = lib.conv(uni, [1, 1, uni.get_shape().as_list()[-1], channel],
                                   w_fn=lib.xavieru_conv_w, b_fn=lib.zero_b, act_fn=g_linear)

                uni = uni + shortcut

            with tf.variable_scope("bigram"):
                bi = net
                with tf.variable_scope("conv1x1_1"):
                    bi = lib.conv(bi, [1, 1, bi.get_shape().as_list()[-1], channel//2],
                                  w_fn=lib.xavieru_conv_w, b_fn=lib.zero_b, act_fn=g_linear)
                    bi = g_ln_act(bi, self.q_len)

                with tf.variable_scope("conv1x2"):
                    bi = lib.conv(bi, [1, 2, bi.get_shape().as_list()[-1], channel],
                                   w_fn=lib.xavieru_conv_w, b_fn=lib.zero_b, act_fn=g_linear)

                bi = bi + shortcut

            with tf.variable_scope("trigram"):
                tri = net
                with tf.variable_scope("conv1x1_1"):
                    tri = lib.conv(tri, [1, 1, tri.get_shape().as_list()[-1], channel//2],
                                   w_fn=lib.xavieru_conv_w, b_fn=lib.zero_b, act_fn=g_linear)
                    tri = g_ln_act(tri, self.q_len)

                with tf.variable_scope("conv1x3"):
                    tri = lib.conv(tri, [1, 3, tri.get_shape().as_list()[-1], channel],
                                   w_fn=lib.xavieru_conv_w, b_fn=lib.zero_b, act_fn=g_linear)
                tri = tri + shortcut

            with tf.variable_scope("maxout"):
                with tf.variable_scope("concat"):
                    net = tf.concat([uni, bi, tri], 2)

                with tf.variable_scope("max_pool"):
                    net = tf.nn.max_pool(net, [1,1,3,1], [1,1,1,1], "VALID")

        return net

    def summarize_tensor(self):
        [tf.summary.histogram(tag_name(w), w) for w in self.weights]

    @property
    def weights(self):
        return [w for w in tf.trainable_variables()
                if self.var_scope_str in w.name]

    @property
    def weights_l1(self):
        return [w for w in tf.trainable_variables()
                if self.block_1_name in w.name]

    @property
    def weights_l2(self):
        return [w for w in tf.trainable_variables()
                if self.block_2_name in  w.name]

    @property
    def weights_l3(self):
        return [w for w in tf.trainable_variables()
                if self.block_3_name in w.name]


class Attention:
    def __init__(self, var_scope_str, batch_size):
        self.var_scope_str = var_scope_str
        self.batch_size = batch_size

    def __call__(self, v, q, keep_prob, q_len):
        # v (batch_size, 14, 14, 2048) g_bn_relu
        # q (batch_size, 27, 1, g_channel) g_ln_act
        with tf.variable_scope(self.var_scope_str):

            with tf.variable_scope("v_reshape"):
                v = tf.reshape(v, [self.batch_size, -1, 1, v.get_shape().as_list()[3]])
                # v (batch_size, 196, 1, 2048)

            """
            v0, q0 = self.co_att(v, q, q_len, 512, "co_attention")

            with tf.variable_scope("v0_expand_dim"):
                v0 = tf.expand_dims(v0, 1)
                v0 = tf.expand_dims(v0, 2)

            with tf.variable_scope("q0_expand_dim"):
                q0 = tf.expand_dims(q0, 1)
                q0 = tf.expand_dims(q0, 2)

            """
            with tf.variable_scope("v0"):
                v0 = tf.reduce_mean(v, 1, True)
                v0 = lib.conv(v0, [1,1,2048,g_channel], act_fn=g_act)
                v0 = tf.nn.dropout(v0, keep_prob, [self.batch_size,1,1,g_channel])

            with tf.variable_scope("q0"):
                q0_list = tf.unstack(q)
                means = []
                for i in xrange(self.batch_size):
                    sliced = q0_list[i][:q_len[i],:,:]
                    means.append(tf.reduce_mean(sliced, 0, True))
                q0 = tf.stack(means)

            with tf.variable_scope("m0"):
                #
                m0 = v0 * q0

            #
            with tf.variable_scope("v1"):
                v1 = self._att(v, m0, keep_prob, scope="att_v1")
                v1 = lib.conv(v1, [1,1,2048,g_channel], act_fn=g_act)
                v1 = tf.nn.dropout(v1, keep_prob, [self.batch_size,1,1,g_channel])

            with tf.variable_scope("q1"):
                q1 = self._att(q, m0, keep_prob, scope="att_q1", q_len=q_len)

            with tf.variable_scope("m1"):
                m1 = v1 * q1 + m0

            #
            with tf.variable_scope("v2"):
                v2 = self._att(v, m1, keep_prob, scope="att_v2")
                v2 = lib.conv(v2, [1,1,2048,g_channel], act_fn=g_act)
                v2 = tf.nn.dropout(v2, keep_prob, [self.batch_size,1,1,g_channel])

            with tf.variable_scope("q2"):
                q2 = self._att(q, m1, keep_prob, scope="att_q2", q_len=q_len)

            with tf.variable_scope("m2"):
                m2 = v2 * q2 + m1

        self.m0 = m0
        self.m1 = m1
        self.m2 = m2

        return m2

    def _att(self, a, b, keep_prob, scope, q_len=None):
        """ b attends to a """
        with tf.variable_scope(scope):
            # kernel trick here?
            with tf.variable_scope("ah"):
                ah = lib.conv(a, [1, 1, a.get_shape().as_list()[-1], g_channel],
                               w_fn=lib.xavieru_conv_w, b_fn=lib.zero_b, act_fn=g_act)
                ah = tf.nn.dropout(ah, keep_prob, [self.batch_size,1,1,g_channel])

            with tf.variable_scope("bh"):
                bh = lib.conv(b, [1, 1, b.get_shape().as_list()[-1], g_channel],
                               w_fn=lib.xavieru_conv_w, b_fn=lib.zero_b, act_fn=g_act)
                bh = tf.nn.dropout(bh, keep_prob, [self.batch_size,1,1,g_channel])

            with tf.variable_scope("alpha"):
                alpha = ah * bh
                with tf.variable_scope("conv1x1"):
                    alpha = lib.conv(alpha, [1,1,g_channel,1],
                                     w_fn=lib.xavieru_conv_w, b_fn=None, act_fn=g_linear)

            if q_len is not None:
                with tf.variable_scope("separate_softmax_attention"):
                    attended_list = []
                    alpha_list = tf.unstack(alpha)
                    a_list = tf.unstack(a)
                    for i in xrange(self.batch_size):
                        sliced_alpha = alpha[i][:q_len[i],:,:]
                        sliced_alpha = lib.softmax(sliced_alpha, 0)

                        sliced_a = a[i][:q_len[i],:,:]
                        with tf.variable_scope("attended"):
                            attended_a = tf.reduce_sum(sliced_a * sliced_alpha, 0, True)
                        attended_list.append(attended_a)
                    attended_a = tf.stack(attended_list, 0)
            else:
                with tf.variable_scope("softmax"):
                    alpha = lib.softmax(alpha, 1)

                with tf.variable_scope("attended"):
                    attended_a = tf.reduce_sum(a * alpha, 1, True)

        return attended_a

    def co_att(self, v, q, q_len, dim, var_scope_str):

        # v (batch_size, 196, 1, 2048)
        # q (batch_size, 27, 1, 256)
        v_dim = v.get_shape().as_list()[-1]
        q_dim = q.get_shape().as_list()[-1]

        with tf.variable_scope(var_scope_str):
            with tf.variable_scope("C"):
                b_c_q = lib.zero_b("bias_q", [q_dim])
                b_c_v = lib.zero_b("bias_v", [v_dim])
                with tf.variable_scope("QW"):
                    QW = lib.conv(q + b_c_q, [1,1,q_dim,v_dim],
                                  b_fn=None, act_fn=g_linear)
                # (b, 27, 1, 2048)
                QW = tf.squeeze(QW, 2)
                # (b, 27, 2048)
                V = tf.squeeze(v + b_c_v)
                # (b, 196, 2048)
                QWV = tf.matmul(QW, V, transpose_b=True)
                # (b, 27, 196)
                C = g_act(QWV)
                # (b, 27, 196)

            with tf.variable_scope("WQ"):
                WQ = lib.conv(q, [1,1,q_dim,dim],
                              b_fn=lib.zero_b, act_fn=g_linear)
                # (b, 27, 1, dim)
                WQ = tf.transpose(tf.squeeze(WQ, 2), [0,2,1])
                # (b, dim, 27)

            with tf.variable_scope("WV"):
                WV = lib.conv(v, [1,1,v_dim,dim],
                              b_fn=lib.zero_b, act_fn=g_linear)
                # (b, 196, 1, dim)
                WV = tf.transpose(tf.squeeze(WV, 2), [0,2,1])
                # (b, dim, 196)

            with tf.variable_scope("H_v"):
                WQC = tf.matmul(WQ, C)
                # (b, dim, 196)
                WV_WQC = tf.add(WV, WQC)
                # (b, dim, 196)
                H_v = g_act(WV_WQC)
                # (b, dim, 196)

            with tf.variable_scope("v_attention"):
                H_v = tf.expand_dims(tf.transpose(H_v, [0,2,1]), 2)
                # (b, 196, 1, dim)
                wH_v = lib.conv(H_v, [1,1,dim,1], act_fn=g_linear)
                wH_v = tf.squeeze(wH_v, [2, 3])
                # (b, 196)
                a_v = lib.softmax(wH_v)
                # (b, 196)

                with tf.variable_scope("attented_v"):
                    v_hat = tf.multiply(tf.squeeze(v, 2), tf.expand_dims(a_v, 2))
                    # (b, 196, v_dim)
                    v_hat = tf.reduce_sum(v_hat, 1)
                    # (b, v_dim)

            with tf.variable_scope("H_q"):
                WVC = tf.matmul(WV, C, transpose_b=True)
                # (b, dim, 27)
                WQ_WVC = tf.add(WQ, WVC)
                # (b, dim, 27)
                H_q = g_act(WQ_WVC)
                # (b, dim, 27)

            with tf.variable_scope("q_attention"):
                H_q = tf.expand_dims(tf.transpose(H_q, [0,2,1]), 2)
                # (b, 27, 1, dim)
                wH_q = lib.conv(H_q, [1,1,dim,1], act_fn=g_linear)
                # (b, 27, 1, 1)
                wH_q = tf.squeeze(wH_q, [2, 3])
                # (b, 27)
                with tf.variable_scope("split_softmax_attention_concat"):
                    q_hat = []
                    for i in xrange(self.batch_size):
                        sliced_q = q[i, :q_len[i], :]
                        # (q_len[i], q_dim)

                        sliced_wH_q = lib.softmax(wH_q[i, :q_len[i]])
                        # (q_len[i])

                        attention = tf.multiply(
                                tf.squeeze(sliced_q, 1), tf.expand_dims(sliced_wH_q, 1))
                        # (q_len[i], q_dim)
                        attention = tf.reduce_sum(attention, 0)
                        # (q_dim)
                        q_hat.append(attention)
                    q_hat = tf.stack(q_hat, 0)
                    # (b, q_dim)

            return v_hat, q_hat

    def summarize_tensor(self):
        [tf.summary.histogram(tag_name(w), w) for w in self.weights]

    @property
    def weights(self):
        return [w for w in tf.trainable_variables()
                if self.var_scope_str in w.name]

class Answer:
    def __init__(self, var_scope_str, batch_size):
        self.var_scope_str = var_scope_str
        self.batch_size = batch_size

    def __call__(self, x, a_vocab_size, keep_prob):
        with tf.variable_scope(self.var_scope_str):
            net = x
            net = tf.squeeze(net, [1,2])

            with tf.variable_scope("logit"):
                fan_in = g_channel
                fan_out = a_vocab_size
                fan_avg = (fan_in + fan_out) / 2
                scaling = tf.sqrt(1.0 / fan_avg)
                net = lib.scaling_dense(net, a_vocab_size, b_fn=lib.zero_b,
                                        act_fn=lib.linear,
                                        scaling=scaling)
                logit = net

        return logit

    def summarize_tensor(self):
        return [tf.summary.histogram(tag_name(w), w) for w in self.weights]

    @property
    def weights(self):
        return [w for w in tf.trainable_variables()
                if self.var_scope_str in w.name]

class VQA:
    def __init__(self, var_scope_str, q_vocab_size, a_vocab_size, batch_size):
        self.var_scope_str = var_scope_str
        self.q_vocab_size = q_vocab_size
        self.a_vocab_size = a_vocab_size
        self.batch_size = batch_size
        self.embed_size = 128

    def forward_graph(self, qid_input=None, v_input=None, q_input=None, q_len_input=None):
        with tf.variable_scope(self.var_scope_str):
            if qid_input is not None:
                qid = qid_input
            else:
                qid = tf.placeholder(tf.int32, [self.batch_size, 1])

            if v_input is not None:
                v_ph = tf.reshape(v_input, [self.batch_size,14,14,2048])
            else:
                v_ph = tf.placeholder(tf.float32, [self.batch_size,14,14,2048], "v_ph")

            if q_input is not None:
                q_ph = q_input
            else:
                q_ph = tf.placeholder(tf.int32, [self.batch_size, None], "q_ph")

            if q_len_input is not None:
                q_len = tf.reshape(q_len_input, [self.batch_size])
            else:
                q_len = tf.placeholder(tf.int32, [self.batch_size])

            keep_prob_ph = tf.placeholder(tf.float32, [], "keep_prob_ph")

            v_model = Visual("visual", self.batch_size)
            v = v_model(v_ph, keep_prob_ph)

            embed = Embed("q_embed", self.q_vocab_size, self.embed_size, self.batch_size)
            q_0 = embed(q_ph)

            q_model = Question("question", self.batch_size)
            q = q_model(q_0, keep_prob_ph, q_len)

            att_model_1 = Attention("attention_1", self.batch_size)
            att_1 = att_model_1(v, q, keep_prob_ph, q_len)

            ans_model = Answer("answer", self.batch_size)
            logit = ans_model(att_1, self.a_vocab_size, keep_prob_ph)
            pred_idx = tf.argmax(logit, 1)
            softmax_logit = lib.softmax(logit)

            #v_model.summarize_tensor()
            #embed.summarize_embed()
            #q_model.summarize_tensor()
            att_model_1.summarize_tensor()

            #ans_model.summarize_tensor()

            regu_vars = []
            regu_vars += [w for w in embed.weights]
            regu_vars += [w for w in v_model.weights]
            regu_vars += [w for w in q_model.weights]
            regu_vars += [w for w in att_model_1.weights]
            regu_vars += [w for w in ans_model.weights]
            regu_vars = [w for w in regu_vars
                    if "bias" not in w.name and "gamma" not in w.name and "beta" not in w.name]

            self.embed_model = embed
            self.q_model = q_model
            self.v_model = v_model
            self.att_model = att_model_1
            self.ans_model = ans_model

            self.regu_vars = regu_vars
            #self.no_zero = no_zero
            self.qid = qid
            self.v_ph = v_ph
            self.q_ph = q_ph
            self.q_len = q_len
            self.logit = logit
            self.pred_idx = pred_idx
            self.softmax_logit = softmax_logit
            self.keep_prob_ph = keep_prob_ph

    def training_graph(self, a_input=None):
        if a_input is not None:
            a_ph = a_input
        else:
            a_ph = tf.placeholder(tf.int64, [self.batch_size, 1], "a_ph")
        with tf.variable_scope("accuracy"):
            correct = tf.equal(self.pred_idx, tf.argmax(a_ph, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.variable_scope("loss"):
            answer_one_hot = tf.squeeze(tf.one_hot(a_ph, self.a_vocab_size), [1])
            logit = self.logit - tf.reduce_max(self.logit, 1, True)
            xent = tf.losses.softmax_cross_entropy(answer_one_hot, logit)
            #xent = tf.losses.hinge_loss(answer_one_hot, self.logit, 1e3)

            """
            xent = tf.nn.nce_loss(tf.transpose(self.ans_model.w_logit),
                                 self.ans_model.b_logit,
                                 a_ph,
                                 self.ans_model.pre_logit,
                                 num_sampled=nce_samples,
                                 num_classes=2000)
            xent = tf.reduce_mean(xent)
            """
            #l2_regu = 1e-4 * tf.reduce_sum(
            #        [tf.nn.l2_loss(w) for w in self.weights if "bias" not in w.name])
            l2_regu = regu_lambda * tf.reduce_sum([tf.nn.l2_loss(w) for w in self.regu_vars])

            #zero_penalty = tf.reduce_sum([tf.exp(-tf.nn.l2_loss(w)) for w in self.no_zero])

            #loss = xent + l2_regu + zero_penalty
            loss = xent + l2_regu

        with tf.variable_scope("optimizer"):
            global_step = tf.Variable(0, trainable=False)
            #learning_rate = base_learning_rate
            learning_rate = tf.train.piecewise_constant(
                    global_step,
                    [800, 30*train_pairs*2//FLAGS.batch_size, 60*train_pairs*2//FLAGS.batch_size],
                    [warm_start_learning_rate, base_learning_rate, 0.1*base_learning_rate, 0.01*base_learning_rate])
            #learning_rate = tf.train.exponential_decay(
            #        learning_rate, global_step, train_pairs//FLAGS.batch_size, 0.95, True)
            #optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1)
            #optimizer = tf.train.AdamOptimizer(learning_rate)
            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9, epsilon=0.1)
            #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
            if train_embed:
                var_list = tf.trainable_variables()
            else:
                var_list = [w for w in tf.trainable_variables()
                        if "embed" not in w.name]
            gvs = optimizer.compute_gradients(loss, var_list)
            #gvs = [(tf.clip_by_norm(g, 5.0, -1), v)
            #        for g,v in gvs if g is not None]
            gvs = [(tf.clip_by_value(g, -0.1, 0.1), v)
                    for g,v in gvs if g is not None]

            if eta is not None:
                new_gvs = []
                for g,v in gvs:
                    if g is not None:
                        try:
                            g = g + tf.random_normal(
                                    g.get_shape(), 0.0,
                                    tf.sqrt(eta/((1.0+tf.cast(global_step, tf.float32))*0.55)))
                        except ValueError:
                            if "embed" in v.name:
                                g.set_shape([16670, self.embed_size])
                            g = g + tf.random_normal(
                                    g.get_shape(), 0.0,
                                    tf.sqrt(eta/((1.0+tf.cast(global_step, tf.float32))*0.55)))
                        new_gvs.append((g,v))
                gvs = new_gvs

            train_step = optimizer.apply_gradients(gvs, global_step)

        tf.summary.scalar("xent", xent)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("l2_regu", l2_regu)
        #tf.summary.scalar("zero_penalty", zero_penalty)
        tf.summary.scalar("total_loss", loss)
        tf.summary.scalar("learning_rate", learning_rate)
        [tf.summary.histogram(tag_name(v, "_grad"), g) for g,v in gvs]

        self.a_ph = a_ph
        self.accuracy = accuracy
        self.xent = xent
        self.l2_regu = l2_regu
        self.loss = loss
        self.global_step = global_step
        self.learning_rate = learning_rate
        self.gvs = gvs
        self.train_step = train_step


    @property
    def weights(self):
        return [w for w in tf.trainable_variables()
                if self.var_scope_str in w.name]

    def pre_start(self, restore_path=None, logdir=None, pretrained_embeddings=None):
        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=300)

        if logdir:
            writer = tf.summary.FileWriter(logdir, sess.graph)
        else:
            logdir = "./" + self.var_scope_str + "_log"
            writer = tf.summary.FileWriter(logdir, sess.graph)
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        if restore_path:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, restore_path)
            print(" [v] restore ckpt %s" % restore_path)
        else:
            #sess.run(tf.global_variables_initializer())
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            if pretrained_embeddings is not None:
                print("[*] use pretrained embeddings")
                embeddings = self.embed_model.embeddings
                # handle pad embedding
                to_assign = tf.placeholder(tf.float32, embeddings.get_shape())
                sess.run(tf.assign(embeddings, to_assign),
                        {to_assign: pretrained_embeddings})

        merged = tf.summary.merge_all()

        self.sess = sess
        self.saver = saver
        self.logdir = logdir
        self.writer = writer
        self.merged = merged

    def train_one_step(self, bottleneck, question, q_len, answer, write=False):
        feed = {
                self.v_ph: bottleneck,
                self.q_ph: question,
                self.a_ph: answer,
                self.q_len: q_len,
                self.keep_prob_ph: 0.5,
        }
        if write:
            _, summary_str, step, acc_value = self.sess.run(
                    [self.train_step, self.merged, self.global_step, self.accuracy], feed)
            self.writer.add_summary(summary_str, step)
            return acc_value
        else:
            self.sess.run(self.train_step, feed)

    def train_with_loader(self, vqa_loader, epoch, from_epochs=0):
        print("train_with_loader from epoch %d" % from_epochs)
        for e in range(from_epochs, from_epochs + epoch):
            acc_sum = 0.0
            for i in xrange(vqa_loader.num_batches_per_epoch):
                qid, bottleneck, question, q_len, answer = vqa_loader.get_example()
                if i % 100 == 0:
                    acc_sum += self.train_one_step(bottleneck, question, q_len, answer, True)
                else:
                    self.train_one_step(bottleneck, question, q_len, answer, False)
            self.saver.save(self.sess, os.path.join(self.logdir, "ckpt"), (e+1))
            print("%s: epoch %d sampled accuracy %f" % (datetime.datetime.now(), (e+1), acc_sum))

    def train_with_pipeline(self, steps_to_save, epoch, from_epochs=0):
        print("train_with_pipeline from epoch %d" % from_epochs)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:
            step = self.sess.run(self.global_step)
            while not coord.should_stop():
                if step % FLAGS.steps_to_summarize == 0:
                    _, __, summary_str, step = self.sess.run(
                            [self.qid, self.train_step, self.merged, self.global_step], {self.keep_prob_ph:.5})
                    self.writer.add_summary(summary_str, step)
                else:
                    self.sess.run(self.train_step, {self.keep_prob_ph:.5})
                step += 1
                if step % steps_to_save == 0:
                    epoch = step // steps_to_save
                    self.saver.save(self.sess, os.path.join(self.logdir, "ckpt"), epoch)
                    print("%s:epoch %d checkpoint saved" % (datetime.datetime.now(), epoch))
        except tf.errors.OutOfRangeError:
            print("Done")
        finally:
            coord.request_stop()
            coord.join(threads)

    def predict_one(self, bottleneck, question, q_len):
        feed = {
                self.v_ph: bottleneck,
                self.q_ph: question,
                self.q_len: q_len,
                self.keep_prob_ph: 1.0,
        }
        return self.sess.run(self.pred_idx, feed)

    def predict_with_loader(self, vqa_loader):
        pred = {}
        while not vqa_loader.test_end():
            qid, bottleneck, question, q_len = vqa_loader.get_test()
            pred[qid] = self.predict_one(bottleneck, question, q_len)
        return pred

    def predict_with_reader_thread(self, rt):
        prediction = {}
        try:
            while True:
                qid, bottleneck, question, q_len, answer = rt.get_example()
                pred = self.predict_one(bottleneck, question, q_len)
                for i, p in zip(qid, pred):
                    prediction[i] = p
        except:
            pass
        return prediction

    def predict_with_pipeline(self):
        print("predict_with_pipeline")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:
            prediction = {}
            while not coord.should_stop():
                qid, pred = self.sess.run([self.qid, self.pred_idx],
                        {self.keep_prob_ph:1.})
                for q, p in zip(qid, pred):
                    prediction[int(q)] = p

        except tf.errors.OutOfRangeError:
            print("Done")
        finally:
            coord.request_stop()
            coord.join(threads)

        return prediction

    def train_with_pipeline_v2(self, batch, steps_to_save, epoch, from_epochs=0):
        print("train_with_pipeline_v2 from epoch %d" % from_epochs)
        qid_q = vqa_processor.wrapper(
                '../data/Questions/v2_OpenEnded_mscoco_train2014_questions.json')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:
            step = self.sess.run(self.global_step)
            while not coord.should_stop():
                # fetch data
                qid, bottleneck, question, q_len, answer = self.sess.run(batch)
                # get question from new preproc
                question = []
                for i in np.reshape(qid, [-1]):
                    question.append(qid_q[i]['encoded_question'])
                # set feed_dict
                feed = {
                        self.qid: qid,
                        self.v_ph: np.reshape(bottleneck, [self.batch_size,13,13,1536]),
                        self.q_ph: question,
                        self.q_len: np.reshape(q_len, [-1]),
                        self.a_ph: answer
                }
                # run
                if step % 100 == 0:
                    _, __, summary_str, step = self.sess.run(
                            [self.qid, self.train_step, self.merged, self.global_step], feed)
                    self.writer.add_summary(summary_str, step)
                else:
                    self.sess.run(self.train_step, feed)
                step += 1
                if step % steps_to_save == 0:
                    epoch = step // steps_to_save
                    self.saver.save(self.sess, os.path.join(self.logdir, "ckpt"), epoch)
                    print("%s:epoch %d checkpoint saved" % (datetime.datetime.now(), epoch))
        except tf.errors.OutOfRangeError:
            print("Done")
        finally:
            coord.request_stop()
            coord.join(threads)

    def train_with_pipeline_v3(self, batch, steps_to_save, epoch, from_epochs=0):
        print("train_with_pipeline_v3 from epoch %d" % from_epochs)
        qid_q = vqa_processor.wrapper(
                '../data/Questions/v2_OpenEnded_mscoco_train2014_questions.json')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        acc = []
        try:
            step = self.sess.run(self.global_step)
            while not coord.should_stop():
                # fetch data
                qid, bottleneck, question, q_len, answer = self.sess.run(batch)
                # get question from new preproc
                question = []
                for i in np.reshape(qid, [-1]):
                    question.append(qid_q[i]['encoded_question'])
                # set feed_dict
                feed = {
                        self.qid: qid,
                        self.v_ph: np.reshape(bottleneck, [self.batch_size,13,13,1536]),
                        self.q_ph: question,
                        self.q_len: np.reshape(q_len, [-1]),
                        self.a_ph: answer,
                        self.keep_prob_ph: 0.5
                }
                # run
                if step % 100 == 0:
                    _, __, summary_str, step, acc_np = self.sess.run(
                            [self.qid, self.train_step, self.merged, self.global_step,
                                self.accuracy], feed)
                    self.writer.add_summary(summary_str, step)
                    acc.append(acc_np)
                else:
                    _, acc_np = self.sess.run([self.train_step, self.accuracy], feed)
                    acc.append(acc_np)
                step += 1
                if step % steps_to_save == 0:
                    print(" [*] accuracy %f" % np.mean(acc))
                    acc = []
                    epoch = step // steps_to_save
                    self.saver.save(self.sess, os.path.join(self.logdir, "ckpt"), epoch)
                    print("%s:epoch %d checkpoint saved" % (datetime.datetime.now(), epoch))
        except tf.errors.OutOfRangeError:
            print("Done")
        finally:
            coord.request_stop()
            coord.join(threads)


    def predict_with_pipeline_v2(self, batch):
        print("predict_with_loader")
        qid_q = vqa_processor.wrapper(
                '../data/Questions/v2_OpenEnded_mscoco_val2014_questions.json',
                pad_to_len=27)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        prediction = {}
        try:
            while not coord.should_stop():
                # fetch data
                qid, bottleneck, question, q_len, answer = self.sess.run(batch)
                # get question from new preproc
                question = []
                for i in np.reshape(qid, [-1]):
                    question.append(qid_q[i]['encoded_question'])
                # set feed_dict
                feed = {
                        self.qid: qid,
                        self.v_ph: np.reshape(bottleneck, [-1,13,13,1536]),
                        self.q_ph: question,
                        self.q_len: np.reshape(q_len, [-1]),
                }
                # run
                qid, pred = self.sess.run([self.qid, self.pred_idx], feed)
                for q, p in zip(qid, pred):
                    prediction[int(q)] = p

        except tf.errors.OutOfRangeError:
            print("Done")
        finally:
            coord.request_stop()
            coord.join(threads)

        return prediction

    def predict_with_pipeline_v3(self, batch):
        print("predict_with_pipeline_v3")
        qid_q = vqa_processor.wrapper(
                '../data/Questions/v2_OpenEnded_mscoco_val2014_questions.json',
                pad_to_len=27)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        prediction = {}
        acc = []
        try:
            while not coord.should_stop():
                # fetch data
                qid, bottleneck, question, q_len, answer = self.sess.run(batch)
                # get question from new preproc
                question = []
                for i in np.reshape(qid, [-1]):
                    question.append(qid_q[i]['encoded_question'])
                # set feed_dict
                feed = {
                        self.qid: qid,
                        self.v_ph: np.reshape(bottleneck, [-1,13,13,1536]),
                        self.q_ph: question,
                        self.q_len: np.reshape(q_len, [-1]),
                        self.a_ph: answer,
                        self.keep_prob_ph: 1.0,
                }
                # run
                qid, pred, acc_np = self.sess.run([self.qid, self.pred_idx, self.accuracy], feed)
                acc.append(acc_np)

        except tf.errors.OutOfRangeError:
            print("Done")
        finally:
            coord.request_stop()
            coord.join(threads)

        print(" [*] accuracy %f" % np.mean(acc))

        return np.mean(acc)


def main(_):
    # tfrecord
    if FLAGS.phase == 'train':
        paired = glob.glob('../data/prepro_v2/vqa_data/train/train_paired-*')
        pseudo_paired = glob.glob('../data/prepro_v2/train/train_pseudo_paired-*')
        tfrecord_list = paired + pseudo_paired
        batch_size = FLAGS.batch_size
        is_train = True
        epochs = None

    elif FLAGS.phase == 'train_all':
        train_paired = glob.glob('../data/prepro_v2/vqa_data/train/train_paired-*')
        train_pseudo_paired = glob.glob('../data/prepro_v2/train/train_pseudo_paired-*')
        val_paired = glob.glob('../data/prepro_v2/vqa_data/val/val_paired-*')
        val_pseudo_paired = glob.glob('../data/prepro_v2/val/val_pseudo_paired-*')
        tfrecord_list = train_paired + train_pseudo_paired + val_paired + val_pseudo_paired
        batch_size = FLAGS.batch_size
        is_train = True
        epochs = None
        global train_pairs
        train_pairs = all_pairs

    elif FLAGS.phase == 'train_sample':
        tfrecord_list = [
                '../data/prepro_v2/vqa_data/train/train_paired-0',
                '../data/prepro_v2/vqa_data/train/train_paired-1',
                ]
        batch_size = FLAGS.batch_size
        is_train = True
        epochs = None
        global train_pairs
        train_pairs = sample_pairs

    elif FLAGS.phase == 'predict_train':
        paired = glob.glob('../data/prepro_v2/vqa_data/train/train_paired-*')
        pseudo_paired = glob.glob('../data/prepro_v2/train/train_pseudo_paired-*')
        tfrecord_list = paired + pseudo_paired
        # (200394 + 23954) * 2 = 448696 = 2 * 2 * 2 * 56087
        batch_size = 8
        is_train = False
        pred_post_name = "_pred_tr"
        val_ann = '../data/Annotations/v2_mscoco_train2014_annotations.json'
        val_ques = '../data/Questions/v2_OpenEnded_mscoco_train2014_questions.json'

    elif FLAGS.phase == 'predict_train_sample':
        tfrecord_list = [
                '../data/prepro_v2/vqa_data/train/train_paired-0',
                '../data/prepro_v2/vqa_data/train/train_paired-1',
                ]
        # 10000 * 2
        batch_size = 50
        is_train = False
        pred_post_name = "_pred_tr_s"
        val_ann = '../data/prepro_v2/sampled_tr_ann_paired.json'
        val_ques = '../data/prepro_v2/sampled_tr_ques_paired.json'

    elif FLAGS.phase == 'predict_val':
        paired = glob.glob('../data/prepro_v2/vqa_data/val/val_paired-*')
        pseudo_paired = glob.glob('../data/prepro_v2/val/val_pseudo_paired-*')
        tfrecord_list = paired + pseudo_paired
        # (95144 + 13162) * 2 = 108306 * 2 = 2 * 2 * 3 * 3 * 11 * 547
        batch_size = 66
        is_train = False
        pred_post_name = "_pred_va"
        val_ann = '../data/Annotations/v2_mscoco_val2014_annotations.json'
        val_ques = '../data/Questions/v2_OpenEnded_mscoco_val2014_questions.json'

    elif FLAGS.phase == 'predict_val_sample':
        tfrecord_list = [
                '../data/prepro_v2/vqa_data/val/val_paired-0',
                '../data/prepro_v2/vqa_data/val/val_paired-1',
                ]
        # 10000 * 2
        batch_size = 50
        is_train = False
        pred_post_name = "_pred_va_s"
        val_ann = '../data/prepro_v2/sampled_va_ann_paired.json'
        val_ques = '../data/prepro_v2/sampled_va_ques_paired.json'

    elif FLAGS.phase == 'predict_dev':
        #tfrecord_list = run and save in the sdb
        tfrecord_list = glob.glob('../data/prepro_v2/dev/dev_pseudo_paired-*')
        # 107394 = 2 x 3 x 7 x 2557
        batch_size = 42
        is_train = False
        pred_post_name = "_pred_dev"
        val_ann = None
        val_ques = None

    elif FLAGS.phase == 'predict_test':
        #tfrecord_list = run and save in the sdb
        #batch_size = check how many examples are there in the test
        is_train = False
        pred_post_name = "_pred_test"
        val_ann = None
        val_ques = None

    else:
        pass

    global v_noise
    global q_noise
    global eta

    if is_train:
        epochs = None
        eta = None
        v_noise = None
        q_noise = None
    else:
        epochs = 1
        eta = None
        v_noise = None
        q_noise = None


    # show record_list
    print("tfrecord_list:\n", tfrecord_list)
    batch = tfrecord_reader.paired_inputs(tfrecord_list, is_train, batch_size//2, epochs)

    # build graph
    vqa = VQA("VQA", q_vocab_size , a_vocab_size, batch_size)
    vqa.forward_graph(batch[0], batch[1], batch[2], batch[3])
    if is_train:
        vqa.training_graph(batch[4])

    # pre-start
    pretrained_embeddings = np.load('embed_16670_128.npy')
    vqa.pre_start(FLAGS.restore_path, pretrained_embeddings=pretrained_embeddings)
    del pretrained_embeddings
    if FLAGS.restore_path is not None:
        from_epochs = int(FLAGS.restore_path.split('-')[-1])
    else:
        from_epochs = 0

    # start training or predicting
    if is_train:
        print("%s: start training" % datetime.datetime.now())
        vqa.train_with_pipeline(train_pairs*2//batch_size, 60, from_epochs)
        print("%s: end training" % datetime.datetime.now())

    else:
        print("%s: start predicting" % datetime.datetime.now())
        qid_pred = vqa.predict_with_pipeline()
        qid_pred = vqa_processor.postprocess(qid_pred, './a_vocab.txt')

        save_name = FLAGS.restore_path + pred_post_name
        vqa_processor.save_prediction(save_name, qid_pred)

        if val_ann is not None:
            vqa_processor.evaluate(save_name, val_ann, val_ques)

        print("%s: end predicting" % datetime.datetime.now())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--steps_to_summarize', type=int, default=100)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
