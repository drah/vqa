from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import argparse
import sys
import numpy as np
import datetime
import glob
import collections
import json
import re
import nltk.tokenize
import spacy
import ipdb

FLAGS = None
#TODO: spacy word vocab, spacy embeddings matrix, tfrecord

train_ques_path = '../Questions/v2_OpenEnded_mscoco_train2014_questions.json'
train_ann_path = '../Annotations/v2_mscoco_train2014_annotations.json'
train_pair_path = '../CompPair/v2_mscoco_train2014_complementary_pairs.json'

val_ques_path = '../Questions/v2_OpenEnded_mscoco_val2014_questions.json'
val_ann_path = '../Annotations/v2_mscoco_val2014_annotations.json'
val_pair_path = '../CompPair/v2_mscoco_val2014_complementary_pairs.json'

dev_ques_path = '../Questions/v2_OpenEnded_mscoco_test-dev2015_questions.json'
test_ques_path = '../Questions/v2_OpenEnded_mscoco_test2015_questions.json'

def manage_ques_ann(ques, ann):
    """
    result[qid] = {image_id, question, answer, answers}
    """
    qid_q = manage_ques(ques)

    result = {}
    anns = ann['annotations']
    for ann in anns:
        result[ann['question_id']] = {
                'image_id': qid_q[ann['question_id']]['image_id'],
                'question': qid_q[ann['question_id']]['question'],
                'answer': ann['multiple_choice_answer'],
                'answers': ann['answers'],
                }
    return result

def manage_ques(ques):
    """
    result[qid] = {image_id, question}
    """
    ques = ques['questions']
    qid_q = {
            q['question_id']:{
                'image_id': q['image_id'], 'question': q['question']
                }
            for q in ques}
    return qid_q

def manage_pairs(pairs, pair_no_pair=False, all_qids=None):
    """
    if pair_no_pair is True, all_qids is used to duplicate the
    questions without pair.
    result[qid] = paired_qid
    """
    print("number of pairs: %d" % len(pairs))
    if pair_no_pair:
        qid_in_pair_dict = {}
        for p in pairs:
            qid_in_pair_dict[p[0]] = 1
            qid_in_pair_dict[p[1]] = 1

        count = 0
        for qid in all_qids:
            if qid_in_pair_dict.get(qid, 0) == 0: # not in
                pairs.append([qid, qid])
                count += 1
        print("number of added self-pair: %d" % count)

    return pairs

def get_pseudo_pairs_from_unpaired(pairs, all_qids=None):
    """
    pseudo_pairs = [[qid1, qid2],...], qid1 and qid2 are not complementary
    """
    print("number of pairs: %d" % len(pairs))
    unpaired = []
    qid_in_pair_dict = {}
    for p in pairs:
        qid_in_pair_dict[p[0]] = 1
        qid_in_pair_dict[p[1]] = 1

    count = 0
    for qid in all_qids:
        if qid_in_pair_dict.get(qid, 0) == 0: # not in
            unpaired.append(qid)
            count += 1
    print("number of unpaired: %d" % count)

    if count % 2 != 0: # add random choice to make divisible by 2
        unpaired.append(int(np.random.choice(unpaired)))

    pseudo_pairs = []
    pair = []
    for qid in unpaired:
        pair.append(qid)
        if len(pair) == 2:
            pseudo_pairs.append(pair)
            pair = []
    print("number of pseudo pairs: %d" % len(pseudo_pairs))

    return pseudo_pairs

def tokenize_q(ques, tokenizer='nltk'):
    """
    Args:
        ques: ques[qid] = {question, ...}
        tokenizer: re, nltk, spacy
    Return:
        ques: ques[qid] = {question, tokenized_question, ..}
    """
    def tokenize(sentence):
        return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
            sentence) if i!='' and i!=' ' and i!='\n'];

    def mcb_tokenize(sentence):
	t_str = sentence.lower()
	for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
	    t_str = re.sub( i, '', t_str)
	for i in [r'\-',r'\/']:
	    t_str = re.sub( i, ' ', t_str)
	q_list = re.sub(r'\?','',t_str.lower()).split(' ')
        q_list = filter(lambda x: len(x) > 0, q_list)
        ipdb.set_trace()
        return q_list

    for qid, content in ques.iteritems():
        question = content['question']
        if tokenizer == 'nltk':
            word_tokens = nltk.tokenize.word_tokenize(str(question).lower())
        elif tokenizer == 'spacy':
            #word_tokens = [token.norm_ for token in params['spacy'](s)]
            word_tokens = mcb_tokenize(question)
        elif tokenizer == 're':
            word_tokens = tokenize(str(question).lower())

        content['tokenized_question'] = word_tokens

    return ques

def get_q_vocab(ques, count_thr=0, insert_unk=False):
    """
    Args:
        ques: ques[qid] = {tokenized_question, ...}
        count_thr: int (not included)
        insert_unk: bool, insert_unk or not
    Return:
        vocab: list of vocab
    """

    counts = {}
    for qid, content in ques.iteritems():
        word_tokens = content['tokenized_question']
        for word in word_tokens:
            counts[word] = counts.get(word, 0) + 1

    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str,cw[:20])))

    total_words = sum(counts.itervalues())
    print('total words:', total_words)
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
            (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
            (bad_count, total_words, bad_count*100.0/total_words))

    if insert_unk:
        print('inserting the special UNK token')
        vocab.append('<UNK>')

    return vocab

def get_top_answers(anns, num_answer=2000):
    """
    anns: anns[qid] = {answer, ...}
    """
    counts = {}
    for qid, ann in anns.iteritems():
        answer = ann['answer']
        counts[answer] = counts.get(answer, 0) + 1

    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print('top answer and their counts:')
    print('\n'.join(map(str,cw[:20])))

    vocab = []
    for i in xrange(num_answer):
        vocab.append(cw[i][1])

    return vocab

def save_vocab(filename, vocab):
    """
    vocab: list
    """
    with open(filename, 'w') as f:
        for v in vocab:
            f.write("%s\n" % v)

def encode_question(ques, q_vocab, pad=True, pad_to_len=26):
    """
    Args:
        ques: ques[qid] = {tokenized_question, ...}
        q_vocab: Vocabulary with encode function, pad_id
        pad: bool
        pad_to_len: int
    Return:
        ques: ques[qid] = {encoded_question, question_len, tokenized_question ...}
    """
    for qid, content in ques.iteritems():
        q_len = len(content['tokenized_question'])
        encoded = q_vocab.encode(content['tokenized_question'])
        padded = encoded + [q_vocab.pad_id] * (pad_to_len - q_len)
        content['question_len'] = q_len
        content['encoded_question'] = padded

    return ques

def encode_answer(anns, a_vocab):
    """
    Args:
        anns: anns[qid] = {answer, ...}
    Return:
        anns: anns[qid] = {encoded_answer, answer, ...}
    """
    for qid, content in anns.iteritems():
        content['encoded_answer'] = a_vocab.encode(content['answer'], do_split=False)
    return anns

class Vocabulary:
    def __init__(self, vocab_filename):

        self._id_to_word = {}
        self._word_to_id = {}
        self._bos = None
        self._eos = None
        self._pad = -1
        self._unk = -1

        with open(vocab_filename, 'r') as f:
            for idx, line in enumerate(f):
                word = line.strip()
                if word == "<UNK>":
                    self._unk = idx
                if word == "<PAD>":
                    self._pad = idx
                self._id_to_word[idx] = word
                self._word_to_id[word] = idx

    @property
    def size(self):
        return len(self._id_to_word)

    @property
    def pad_id(self):
        return self._pad

    @property
    def unk_id(self):
        return self._unk

    @property
    def bos_id(self):
        return self._bos

    @property
    def eos_id(self):
        return self._eos

    def word_to_id(self, word):
        return self._word_to_id.get(word, self._unk)

    def id_to_word(self, idx):
        if idx >= self.size:
            return '<OOR>'
        return self._id_to_word.get(idx)

    def encode(self, source, do_split=True):
        if type(source) is list:
            return [self.word_to_id(word) for word in source]
        else:
            return [self.word_to_id(source)]

    def decode(self, ids):
        ids = [ids] if type(ids) is not list else ids
        return ' '.join([self.id_to_word(idx) for idx in ids])

def get_q_max_len(ques):
    max_len = 0
    for qid, content in ques.iteritems():
        if len(content['tokenized_question']) > max_len:
            max_len = len(content['tokenized_question'])
    return max_len

def to_tfrecord(record, record_dir, record_name):

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    shard_count = 0
    shard_name = record_name + "-%d" % shard_count
    shard_path = os.path.join(record_dir, shard_name)
    writer = tf.python_io.TFRecordWriter(shard_path)
    print("%s: write %s start" % (datetime.datetime.now(), shard_path))

    num_ques = len(record.keys())
    i = 0
    for qid, content in record.iteritems():

      features = {}

      bottleneck_path = os.path.join(
              "../Images/incep_res_v2_bottleneck_480/",
              str(content['image_id']).zfill(12)+".npy")
      bottleneck = np.load(bottleneck_path).reshape([-1]).tolist()
      features['bottleneck'] = _float_feature(bottleneck)

      features['question_id'] = _int64_feature([qid])
      features['question'] = _int64_feature(content['encoded_question'])
      features['question_len'] = _int64_feature([content['question_len']])
      if content.has_key('encoded_answer'):
          features['answer'] = _int64_feature(content['encoded_answer'])

      example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(example.SerializeToString())
      i+=1
      print("[%d/%d] records writen" % (i, num_ques), end='\r')
      #sys.stdout.flush()

      # handle shard
      if i % 10000 == 9999:
          writer.close()
          shard_count += 1
          shard_name = record_name + "-%d" % shard_count
          shard_path = os.path.join(record_dir, shard_name)
          writer = tf.python_io.TFRecordWriter(shard_path)
          print("\n%s: write %s start" % (datetime.datetime.now(), shard_path))

    writer.close()
    print("\n%s: write_tfrecord finished" % datetime.datetime.now())

def to_paired_tfrecord(record, record_dir, record_name, pairs, run_count=None):

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    shard_count = 0
    shard_name = record_name + "-%d" % shard_count
    shard_path = os.path.join(record_dir, shard_name)
    writer = tf.python_io.TFRecordWriter(shard_path)
    print("%s: write %s start" % (datetime.datetime.now(), shard_path))

    num_pairs = len(pairs)
    i = 0
    for qid1, qid2 in pairs:
      content1 = record[qid1]

      features = {}

      bottleneck_path = os.path.join(
              "../Images/resnet_v2_bottleneck_448/",
              str(content1['image_id']).zfill(12)+".npy")
      bottleneck_1 = np.load(bottleneck_path).reshape([-1]).tolist()
      features['bottleneck1'] = _float_feature(bottleneck_1)

      features['question_id1'] = _int64_feature([qid1])
      features['question1'] = _int64_feature(content1['encoded_question'])
      features['question_len1'] = _int64_feature([content1['question_len']])
      if content1.has_key('encoded_answer'):
          features['answer1'] = _int64_feature(content1['encoded_answer'])

      content2 = record[qid2]

      bottleneck_path = os.path.join(
              "../Images/resnet_v2_bottleneck_448/",
              str(content2['image_id']).zfill(12)+".npy")
      bottleneck_2 = np.load(bottleneck_path).reshape([-1]).tolist()
      features['bottleneck2'] = _float_feature(bottleneck_2)

      features['question_id2'] = _int64_feature([qid2])
      features['question2'] = _int64_feature(content2['encoded_question'])
      features['question_len2'] = _int64_feature([content2['question_len']])
      if content2.has_key('encoded_answer'):
          features['answer2'] = _int64_feature(content2['encoded_answer'])

      example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(example.SerializeToString())
      i+=1
      if i % 100 == 0:
          print("[%d/%d] records writen" % (i, num_pairs), end='\r')
          sys.stdout.flush()

      # this is for creating sampled shard
      if i == run_count:
          break

      # handle shard
      if i % 5000 == 0:
          writer.close()
          shard_count += 1
          shard_name = record_name + "-%d" % shard_count
          shard_path = os.path.join(record_dir, shard_name)
          writer = tf.python_io.TFRecordWriter(shard_path)
          print("\n%s: write %s start" % (datetime.datetime.now(), shard_path))

    writer.close()
    print("\n%s: write_tfrecord finished" % datetime.datetime.now())

# the followings are some functions to run to get the processed tfrecord

# add this to predict dev without pipeline... 2017/06/21
def dev_wrapper(batch_size):
    # should check how many examples are in dev, set appropriate batch_size, or add pad
    q_vocab = Vocabulary('q_vocab_16670_128.txt')

    ques = json.load(open(dev_ques_path, 'r'))
    data = manage_ques(ques)
    data = tokenize_q(data, tokenizer='re')
    data = encode_question(data, q_vocab, pad=True, pad_to_len=27)
    del q_vocab
    del ques

    ret_qid, ret_v, ret_q, ret_q_len = [], [], [], []
    count = 0
    for qid, content in data.iteritems():
        ret_qid.append([qid])
        bottleneck_path = os.path.join(
                "../Images/resnet_v2_bottleneck_448/",
                str(content['image_id']).zfill(12)+".npy")
        bottleneck = np.load(bottleneck_path).reshape([-1])
        ret_v.append(bottleneck)
        ret_q.append(content['encoded_question'])
        ret_q_len.append(content['question_len'])
        count += 1
        if count == batch_size:
            yield ret_qid, ret_v, ret_q, ret_q_len
            ret_qid, ret_v, ret_q, ret_q_len = [], [], [], []
            count = 0

# add this to run with word2vec pretrained embeddings
def wrapper(ques_path, ann_path=None, tokenizer='re', pad_to_len=26):
    q_vocab = Vocabulary('q_vocab_word2vec.txt')
    a_vocab = Vocabulary('a_vocab.txt')
    if ann_path is not None:
        ques = json.load(open(ques_path, 'r'))
        ann = json.load(open(ann_path, 'r'))
        data = manage_ques_ann(ques, ann)
        data = tokenize_q(data, tokenizer)
        data = encode_question(data, q_vocab, pad=True, pad_to_len=pad_to_len)
        data = encode_answer(data, a_vocab)
    else:
        ques = json.load(open(ques_path, 'r'))
        data = manage_ques(ques)
        data = tokenize_q(data, tokenizer)
        data = encode_question(data, q_vocab, pad=True, pad_to_len=pad_to_len)

    return data
    # used_part is only encoded_question here, because others are provided with pipeline
    used_part = {}
    for qid, content in data.iteritems():
        used_part[qid] = {'encoded_question': content['encoded_question']}

    return used_part

def run_with_word2vec():
    q_vocab = Vocabulary('q_vocab_word2vec.txt')
    a_vocab = Vocabulary('a_vocab.txt')

    ques = json.load(open(train_ques_path, 'r'))
    ann = json.load(open(train_ann_path, 'r'))
    data = manage_ques_ann(ques, ann)
    data = tokenize_q(data, tokenizer='re')
    data = encode_question(data, q_vocab, pad=True, pad_to_len=27)
    data = encode_answer(data, a_vocab)

    to_tfrecord(data, './train', 'train')
    del ques
    del ann
    del data

    ques = json.load(open(val_ques_path, 'r'))
    ann = json.load(open(val_ann_path, 'r'))
    data = manage_ques_ann(ques, ann)
    data = tokenize_q(data, tokenizer='re')
    data = encode_question(data, q_vocab, pad=True, pad_to_len=27)
    data = encode_answer(data, a_vocab)

    to_tfrecord(data, './val', 'val')


def run_with_word2vec_pair(phase):
    q_vocab = Vocabulary('q_vocab_word2vec.txt')
    a_vocab = Vocabulary('a_vocab.txt')

    if phase == 'train':
        ques = json.load(open(train_ques_path, 'r'))
        ann = json.load(open(train_ann_path, 'r'))
        data = manage_ques_ann(ques, ann)
        data = tokenize_q(data, tokenizer='re')
        data = encode_question(data, q_vocab, pad=True, pad_to_len=27)
        data = encode_answer(data, a_vocab)
        pairs = json.load(open(train_pair_path, 'r'))
        pseudo_pairs = get_pseudo_pairs_from_unpaired(pairs, data.keys())
        to_paired_tfrecord(data, './train', 'train_paired', pairs, run_count=None)
        to_paired_tfrecord(data, './train', 'train_pseudo_paired', pseudo_pairs, run_count=None)

    elif phase == 'val':
        ques = json.load(open(val_ques_path, 'r'))
        ann = json.load(open(val_ann_path, 'r'))
        data = manage_ques_ann(ques, ann)
        data = tokenize_q(data, tokenizer='re')
        data = encode_question(data, q_vocab, pad=True, pad_to_len=27)
        data = encode_answer(data, a_vocab)
        pairs = json.load(open(val_pair_path, 'r'))
        pseudo_pairs = get_pseudo_pairs_from_unpaired(pairs, data.keys())
        to_paired_tfrecord(data, './val', 'val_paired', pairs, run_count=None)
        to_paired_tfrecord(data, './val', 'val_pseudo_paired', pseudo_pairs, run_count=None)

    elif phase == 'dev':
        ques = json.load(open(dev_ques_path, 'r'))
        data = manage_ques(ques)
        data = tokenize_q(data, tokenizer='re')
        data = encode_question(data, q_vocab, pad=True, pad_to_len=27)
        pseudo_pairs = get_pseudo_pairs_from_unpaired([], data.keys())
        to_paired_tfrecord(data, './dev', 'dev_pseudo_paired', pseudo_pairs, run_count=None)

    elif phase == 'test':
        ques = json.load(open(test_ques_path, 'r'))
        data = manage_ques(ques)
        data = tokenize_q(data, tokenizer='re')
        data = encode_question(data, q_vocab, pad=True, pad_to_len=27)
        pseudo_pairs = get_pseudo_pairs_from_unpaired([], data.keys())
        to_paired_tfrecord(data, './test', 'test_pseudo_paired', pseudo_pairs, run_count=None)

    elif phase == 'train_sample':
        n_samples = 10000 # would be 20000 samples if rolled out
        pairs = json.load(open(train_pair_path, 'r'))
        to_paired_tfrecord(data, './train', 'sampled_train_paired', pairs, run_count=n_samples)
        # handle q,a subset
        sub_qid_dict = {}
        for qid1,qid2 in pairs[:n_samples]:
            sub_qid_dict[qid1] = 1
            sub_qid_dict[qid2] = 1
        ques['questions'] = [q for q in ques['questions']
                if sub_qid_dict.get(q['question_id'],-1)!=-1]
        json.dump(ques, open('sampled_tr_ques_paired.json', 'w'))
        ann['annotations'] = [a for a in ann['annotations']
                if sub_qid_dict.get(a['question_id'],-1)!=-1]
        json.dump(ann, open('sampled_tr_ann_paired.json', 'w'))

    elif phase == 'val_sample':
        n_samples = 10000 # would be 20000 samples if rolled out
        pairs = json.load(open(val_pair_path, 'r'))
        to_paired_tfrecord(data, './val', 'sampled_val_paired', pairs, run_count=n_samples)
        # handle q,a subset
        sub_qid_dict = {}
        for qid1,qid2 in pairs[:n_samples]:
            sub_qid_dict[qid1] = 1
            sub_qid_dict[qid2] = 1
        ques['questions'] = [q for q in ques['questions']
                if sub_qid_dict.get(q['question_id'],-1)!=-1]
        json.dump(ques, open('sampled_va_ques_paired.json', 'w'))
        ann['annotations'] = [a for a in ann['annotations']
                if sub_qid_dict.get(a['question_id'],-1)!=-1]
        json.dump(ann, open('sampled_va_ann_paired.json', 'w'))

def run():
    train_ques = json.load(open(train_ques_path, 'r'))
    train_ann = json.load(open(train_ann_path, 'r'))
    train = manage_ques_ann(train_ques, train_ann)
    train = tokenize_q(train, tokenizer='nltk')

    q_vocab = get_q_vocab(train, count_thr=1, insert_unk=True)
    save_vocab('q_vocab.txt', q_vocab)

    a_vocab = get_top_answers(train, 2000)
    save_vocab('a_vocab.txt', a_vocab)

    q_vocab = Vocabulary('q_vocab.txt')
    a_vocab = Vocabulary('a_vocab.txt')
    train = encode_question(train, q_vocab, pad=True, pad_to_len=26)
    train = encode_answer(train, a_vocab)
    to_tfrecord(train, './train', 'train')

    val_ques = json.load(open(val_ques_path, 'r'))
    val_ann = json.load(open(val_ann_path, 'r'))
    val = manage_ques_ann(val_ques, val_ann)
    val = tokenize_q(val, tokenizer='nltk')
    val = encode_question(val, q_vocab, pad=True, pad_to_len=26)
    val = encode_answer(val, a_vocab)
    to_tfrecord(val, './val', 'val')

    dev_ques = json.load(open(dev_ques_path, 'r'))
    dev = manage_ques(dev_ques)
    dev = tokenize_q(dev, tokenizer='nltk')
    dev = encode_question(dev, q_vocab, pad=True, pad_to_len=26)
    to_tfrecord(dev, './dev', 'dev')

    test_ques = json.load(open(test_ques_path, 'r'))
    test = manage_ques(test_ques)
    test = tokenize_q(test, tokenizer='nltk')
    test = encode_question(test, q_vocab, pad=True, pad_to_len=26)
    to_tfrecord(test, './test', 'test')

# the followings are for postprocess

def postprocess(qid_pred, a_vocab_path):
    a_vocab = Vocabulary(a_vocab_path)
    for q, p in qid_pred.iteritems():
        qid_pred[q] = a_vocab.decode(p)
    return qid_pred

def save_prediction(save_path, qid_pred):
    result = [{"question_id": int(qid), "answer": pred}
              for qid, pred in qid_pred.iteritems()]
    json.dump(result, open(save_path, 'w'))

def evaluate(predicted_json_path, ann_path, ques_path):
    """revised from official evaluation code

    Args:
      result_path: predicted result in json format.
      ann_path: annotation_file path.
      ques_path: question_file path.
      result_dir_path: if given, save the evalutation result to the dir path.

    """
    from vqa import VQA
    from vqaEval import VQAEval

    vqa = VQA(ann_path, ques_path)
    result = vqa.loadRes(predicted_json_path, ques_path)
    vqa_eval = VQAEval(vqa, result, n=2)
    vqa_eval.evaluate()
    print("\nOverall Accuracy is: %.02f" % (vqa_eval.accuracy['overall']))
    print("Per Question Type Accuracy is the following:")
    for quesType in vqa_eval.accuracy['perQuestionType']:
      print("%s: %.02f" %(quesType, vqa_eval.accuracy['perQuestionType'][quesType]))
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqa_eval.accuracy['perAnswerType']:
      print("%s: %.02f" %(ansType, vqa_eval.accuracy['perAnswerType'][ansType]))

    result_dir_path = predicted_json_path + "_eval"
    if result_dir_path is not None:
      if not os.path.exists(result_dir_path):
        os.makedirs(result_dir_path)
      json.dump(vqa_eval.accuracy, open(os.path.join(result_dir_path, 'accuracy'), 'w'))
      json.dump(vqa_eval.evalQA, open(os.path.join(result_dir_path, 'evalQA'), 'w'))
      json.dump(vqa_eval.evalQuesType, open(os.path.join(result_dir_path, 'evalQuesType'), 'w'))
      json.dump(vqa_eval.evalAnsType, open(os.path.join(result_dir_path, 'evalAnsType'),  'w'))
