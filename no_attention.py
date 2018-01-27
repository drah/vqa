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

g_channel = 600
g_linear = lib.linear
g_act = lib.tanh

g_bn = lib.batch_norm
g_bn_linear = lib.bn_linear
g_bn_act = lib.bn_tanh
warm_start_learning_rate = 1e-1
base_learning_rate = 1e-1

# regularizations
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

        return net

    def summarize_tensor(self):
        [tf.summary.histogram(tag_name(w), w) for w in self.weights]

    @property
    def weights(self):
        return [w for w in tf.trainable_variables()
                if self.var_scope_str in w.name]

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
                m0 = v0 * q0

        self.m0 = m0

        return m0

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
            l2_regu = regu_lambda * tf.reduce_sum([tf.nn.l2_loss(w) for w in self.regu_vars])
            loss = xent + l2_regu

        with tf.variable_scope("optimizer"):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.piecewise_constant(
                    global_step,
                    [800, 30*train_pairs*2//FLAGS.batch_size, 60*train_pairs*2//FLAGS.batch_size],
                    [warm_start_learning_rate, base_learning_rate, 0.1*base_learning_rate, 0.01*base_learning_rate])
            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9, epsilon=0.1)

            if train_embed:
                var_list = tf.trainable_variables()
            else:
                var_list = [w for w in tf.trainable_variables()
                        if "embed" not in w.name]

            gvs = optimizer.compute_gradients(loss, var_list)
            gvs = [(tf.clip_by_value(g, -0.1, 0.1), v)
                    for g,v in gvs if g is not None]

            train_step = optimizer.apply_gradients(gvs, global_step)

        tf.summary.scalar("xent", xent)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("l2_regu", l2_regu)
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

    def train_with_pipeline(self, steps_to_save, steps_to_summarize, epoch, from_epochs=0):
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
                    print(step)
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

def main(_):
    # tfrecord
    if FLAGS.phase == 'train':
        tfrecord_list = glob.glob(os.path.join(FLAGS.data_dir, 'train/*'))
        batch_size = FLAGS.batch_size
        is_train = True
        epochs = None

    elif FLAGS.phase == 'predict_train':
        # (200394 + 23954) * 2 = 448696 = 2 * 2 * 2 * 56087
        tfrecord_list = glob.glob(os.path.join(FLAGS.data_dir, 'train/*'))
        batch_size = 8
        is_train = False
        pred_post_name = "_pred_tr"
        val_ann = os.path.join(FLAGS.data_dir, 'json/v2_mscoco_train2014_annotations.json')
        val_ques = os.path.join(FLAGS.data_dir, 'json/v2_OpenEnded_mscoco_train2014_questions.json')

    elif FLAGS.phase == 'predict_val':
        # (95144 + 13162) * 2 = 108306 * 2 = 2 * 2 * 3 * 3 * 11 * 547
        tfrecord_list = glob.glob(os.path.join(FLAGS.data_dir, 'val/*'))
        batch_size = 66
        is_train = False
        pred_post_name = "_pred_va"
        val_ann = os.path.join(FLAGS.data_dir, 'json/v2_mscoco_val2014_annotations.json')
        val_ques = os.path.join(FLAGS.data_dir, 'json/v2_OpenEnded_mscoco_val2014_questions.json')

    elif FLAGS.phase == 'predict_dev':
        # 107394 = 2 x 3 x 7 x 2557
        tfrecord_list = glob.glob(os.path.join(FLAGS.data_dir, 'dev/*'))
        batch_size = 42
        is_train = False
        pred_post_name = "_pred_dev"
        val_ann = None
        val_ques = None

    elif FLAGS.phase == 'predict_test':
        tfrecord_list = glob.glob(os.path.join(FLAGS.data_dir, 'test/*'))
        is_train = False
        pred_post_name = "_pred_test"
        val_ann = None
        val_ques = None

    else:
        pass

    if is_train:
        epochs = None
    else:
        epochs = 1

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
        vqa.train_with_pipeline(
                FLAGS.steps_to_save, FLAGS.steps_to_summarize, FLAGS.epochs, FLAGS.from_epoch)
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
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--steps_to_save', type=int, default=10000//50)
    parser.add_argument('--steps_to_summarize', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--from_epoch', type=int, default=0)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
