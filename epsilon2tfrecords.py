"""Generating TFRecords of Epsilon dataset for paper:
Haddadpour, F.,  Kamani, M.M., Mahdavi, M., & Cadambe, V.
"Local SGD with Periodic Averaging: Tighter Analysis and Adaptive Synchronization"
 Advances in Neural Information Processing. 2019.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_svmlight_file
from tqdm import tqdm

"""
__author__ = "Mohammad Mahdi Kamani"
__copyright__ = "Copyright 2019, Mohammad Mahdi Kamani"

__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Madhi Kamani"
__status__ = "Prototype"
"""

FILENAMES={
  'train': 'epsilon_normalized.bz2' ,
  'test':  'epsilon_normalized.t.bz2'
}

def _get_dense_tensor(tensor):
    if 'sparse' in str(type(tensor)):
        return tensor.toarray()
    elif 'numpy' in str(type(tensor)):
        return tensor


def _correct_binary_labels(labels, is_01_classes=True):
    classes = set(labels)

    if -1 in classes and is_01_classes:
        labels[labels == -1] = 0
    return labels


class Epsilon_or_RCV1(object):
  def __init__(self, file_path):
    # load dataset.
    dataset = load_svmlight_file(file_path)
    self.features, self.labels = self._get_images_and_labels(dataset)

  def _get_images_and_labels(self, data):
    features, labels = data

    features = _get_dense_tensor(features)
    labels = _get_dense_tensor(labels)
    labels = _correct_binary_labels(labels)
    return features, labels

  def __len__(self):
    return self.features.shape[0]

  def __iter__(self):
    idxs = list(range(self.__len__()))
    for k in idxs:
      yield [self.features[k], self.labels[k]]

  def get_data(self):
    return self.__iter__()

  def size(self):
    return self.__len__()


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(dataset, output_file, subset):
  """Converts a file to TFRecords."""
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    data = dataset.features
    labels = dataset.labels

    num_entries_in_batch = len(labels)
    for i in tqdm(range(num_entries_in_batch)):
      example = tf.train.Example(features=tf.train.Features(
          feature={
              'feature': _bytes_feature(data[i].tobytes()),
              'label': _float_feature(labels[i]),
          }))
      record_writer.write(example.SerializeToString())


def main(params):
  modes = ['train', 'test']
  for mode in modes:
    output_file = os.path.join(params.data_dir, mode + '.tfrecords')
    try:
      os.remove(output_file)
    except OSError:
      pass
    # Convert to tf.train.Example and write the to TFRecords.
    filepath = os.path.join(params.data_dir, FILENAMES[mode])
    print('Loading dataset {}'.format(mode))
    dataset = Epsilon_or_RCV1(filepath)
    print('Finished loading!')
    print('Generating TFrecords for {}'.format(mode))
    convert_to_tfrecord(dataset, output_file, mode)
  print('Done!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      default='',
      help='Directory to download and extract CIFAR-10 to.')
  parser.add_argument(
      '--dataset',
      type=str,
      default='cifar10',
      choices=['cifar10','cifar100'],
      help='The dataset to transfer to TFRecords')


  args = parser.parse_args()
  if args.data_dir:
    tf.gfile.MkDir(args.data_dir)

  main(args)