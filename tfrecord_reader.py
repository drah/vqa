from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from datetime import datetime
import argparse
import sys
import os.path
import glob
import pdb

FLAGS = None
LEN_BOTTLENECK = 14*14*2048
LEN_QUESTION = 27

def inputs(tfrecord_list, has_answer, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
        tfrecord_list: a list of tfrecords
        has_answer: bool to specify having answer or not
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
           train forever.
    Returns:
        qid, bottleneck, question, q_len, answer
    """

    with tf.name_scope('input'):
      filename_queue = tf.train.string_input_producer(
          tfrecord_list, num_epochs=num_epochs)

      qid, bottleneck, question, q_len, answer = read_decode(
              filename_queue, has_answer)

      single_example = [qid, bottleneck, question, q_len, answer]
      batch_example = tf.train.shuffle_batch(
          single_example, batch_size=batch_size, num_threads=2,
          capacity=2048 + 3 * batch_size,
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue=1024)
    return batch_example

def read_decode(filename_queue, has_answer):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    field = {}
    field['question_id'] = tf.FixedLenFeature([1], tf.int64)
    field['bottleneck'] = tf.FixedLenFeature([LEN_BOTTLENECK], tf.float32)
    field['question'] = tf.FixedLenFeature([LEN_QUESTION], tf.int64)
    field['question_len'] = tf.FixedLenFeature([1], tf.int64)
    if has_answer:
        field['answer'] = tf.FixedLenFeature([1], tf.int64)

    features = tf.parse_single_example(serialized_example, features=field)

    qid = tf.cast(features['question_id'], tf.int32)
    bottleneck = features['bottleneck']
    question = tf.cast(features['question'], tf.int32)
    q_len = tf.cast(features['question_len'], tf.int32)
    if has_answer:
        answer = tf.cast(features['answer'], tf.int32)

    if has_answer:
        return qid, bottleneck, question, q_len, answer
    else:
        return qid, bottleneck, question, q_len

def paired_inputs(tfrecord_list, has_answer, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
        tfrecord_list: a list of tfrecords
        has_answer: bool to specify having answer or not
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
           train forever.
    Returns:
        qid, bottleneck, question, q_len, answer
    """

    with tf.name_scope('input'):
      filename_queue = tf.train.string_input_producer(
          tfrecord_list, num_epochs=num_epochs)

      feature_dict = read_decode_paired(filename_queue, has_answer)

      if has_answer:
          single_example = [
                  feature_dict['question_id1'],
                  feature_dict['bottleneck1'],
                  feature_dict['question1'],
                  feature_dict['question_len1'],
                  feature_dict['answer1'],
                  feature_dict['question_id2'],
                  feature_dict['bottleneck2'],
                  feature_dict['question2'],
                  feature_dict['question_len2'],
                  feature_dict['answer2'],
                  ]
      else:
          single_example = [
                  feature_dict['question_id1'],
                  feature_dict['bottleneck1'],
                  feature_dict['question1'],
                  feature_dict['question_len1'],
                  feature_dict['question_id2'],
                  feature_dict['bottleneck2'],
                  feature_dict['question2'],
                  feature_dict['question_len2'],
                  ]

      batch_example = tf.train.shuffle_batch(
          single_example, batch_size=batch_size, num_threads=4,
          capacity=1024 + 3 * batch_size,
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue=512)

      if has_answer:
          concat = [
                  tf.concat([tf.cast(batch_example[0],tf.int32),tf.cast(batch_example[5],tf.int32)],0),
                  tf.concat([batch_example[1],batch_example[6]],0),
                  tf.concat([tf.cast(batch_example[2],tf.int32),tf.cast(batch_example[7],tf.int32)],0),
                  tf.concat([tf.cast(batch_example[3],tf.int32),tf.cast(batch_example[8],tf.int32)],0),
                  tf.concat([tf.cast(batch_example[4],tf.int32),tf.cast(batch_example[9],tf.int32)],0),
                  ]
      else: # wouldn't be used?
          concat = [
                  tf.concat([tf.cast(batch_example[0],tf.int32),tf.cast(batch_example[4],tf.int32)],0),
                  tf.concat([batch_example[1],batch_example[5]],0),
                  tf.concat([tf.cast(batch_example[2],tf.int32),tf.cast(batch_example[6],tf.int32)],0),
                  tf.concat([tf.cast(batch_example[3],tf.int32),tf.cast(batch_example[7],tf.int32)],0),
                  ]

    return concat

def read_decode_paired(filename_queue, has_answer):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    field = {}
    field['question_id1'] = tf.FixedLenFeature([1], tf.int64)
    field['bottleneck1'] = tf.FixedLenFeature([LEN_BOTTLENECK], tf.float32)
    field['question1'] = tf.FixedLenFeature([LEN_QUESTION], tf.int64)
    field['question_len1'] = tf.FixedLenFeature([1], tf.int64)
    if has_answer:
        field['answer1'] = tf.FixedLenFeature([1], tf.int64)
    field['question_id2'] = tf.FixedLenFeature([1], tf.int64)
    field['bottleneck2'] = tf.FixedLenFeature([LEN_BOTTLENECK], tf.float32)
    field['question2'] = tf.FixedLenFeature([LEN_QUESTION], tf.int64)
    field['question_len2'] = tf.FixedLenFeature([1], tf.int64)
    if has_answer:
        field['answer2'] = tf.FixedLenFeature([1], tf.int64)

    features = tf.parse_single_example(serialized_example, features=field)

    return features
    """
    qid = tf.cast(features['question_id'], tf.int32)
    bottleneck = features['bottleneck']
    question = tf.cast(features['question'], tf.int32)
    q_len = tf.cast(features['question_len'], tf.int32)
    if has_answer:
        answer = tf.cast(features['answer'], tf.int32)
    """
