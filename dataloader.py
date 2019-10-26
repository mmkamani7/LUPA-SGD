"""Implementation of data loader for training and testing
Haddadpour, F.,  Kamani, M.M., Mahdavi, M., & Cadambe, V.
"Local SGD with Periodic Averaging: Tighter Analysis and Adaptive Synchronization"
 Advances in Neural Information Processing. 2019.
"""

import os
import numpy as np
import sklearn.decomposition

import tensorflow as tf

"""
__author__ = "Mohammad Mahdi Kamani"
__copyright__ = "Copyright 2019, Mohammad Mahdi Kamani"

__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Madhi Kamani"
__status__ = "Prototype"
"""

class Dataset():
  def __init__(self, data_dir, num_shards, subset='train'):
    self.data_dir = data_dir
    self.num_shards = num_shards
    self.subset = subset

  def get_filenames(self):
    if self.subset in ['train', 'validation', 'test']:
      return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def parser(self):
    return

  def make_batch(self, batch_size):
    feature_shards = [[] for i in range(self.num_shards)]
    label_shards = [[] for i in range(self.num_shards)]

    filenames = self.get_filenames()
    dataset = tf.data.TFRecordDataset(filenames)

    for device_id in range(self.num_shards):
      if self.subset == 'train':
        dataset = tf.data.TFRecordDataset(filenames)
        d0 = dataset.repeat()
        # Parse records.

        d0 = d0.map(
          self.parser, num_parallel_calls=int(batch_size / self.num_shards))
        # Potentially shuffle records.
        min_queue_examples = int(
          self.num_examples_per_epoch(self.subset) * 0.4 / self.num_shards)
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        d0 = d0.shuffle(buffer_size=min_queue_examples + int(3 * batch_size / self.num_shards))

        # Batch it up.
        d0 = d0.batch(int(batch_size / self.num_shards))
        iterator0 = d0.make_one_shot_iterator()
        X_batch, y_batch = iterator0.get_next()

      elif self.subset == 'test':
        d = dataset.repeat()
        d = d.map(self.parser, num_parallel_calls=batch_size)
        d = d.batch(batch_size)
        iterator = d.make_one_shot_iterator()
        X_batch, y_batch = iterator.get_next()


      feature_shards[device_id] = X_batch
      label_shards[device_id] = y_batch

    return feature_shards, label_shards

class MnistDataset(Dataset):

  def __init__(self, data_dir, num_shards, subset='train', redundancy=0.0):
    super(MnistDataset, self).__init__(
      data_dir,
      num_shards,
      subset,
      redundancy
    )
    self.HEIGHT = 28
    self.WIDTH = 28
    self.NUM_CLASSES = 10

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['data'], tf.uint8)
    image = tf.cast(image, tf.float32) / 128.0 - 1
    image.set_shape([self.HEIGHT * self.WIDTH ])
    
    label = tf.one_hot(tf.cast(features['label'], tf.int32), self.NUM_CLASSES)
    return image, label

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 60000
    elif subset == 'eval':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)



class AdultDataset(Dataset):

  def __init__(self, data_dir, num_shards, subset='train', redundancy=0.0):
    super(AdultDataset, self).__init__(
      data_dir,
      num_shards,
      subset,
      redundancy
    )

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    data = tf.decode_raw(features['data'], tf.int64)
    data = tf.cast(tf.reshape(data,[14]), tf.float32)
    
    label = tf.cast(features['label'], tf.int32)
    return data, label

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 30162
    elif subset == 'eval':
      return 15060
    else:
      raise ValueError('Invalid data subset "%s"' % subset)


class EpsilonDataset(Dataset):

  def __init__(self, data_dir, num_shards, subset='train', redundancy=0.0):
    super(EpsilonDataset, self).__init__(
      data_dir,
      num_shards,
      subset,
      redundancy
    )

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'feature': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.float32),
        })
    data = tf.decode_raw(features['feature'], tf.float64)
    data = tf.cast(tf.reshape(data,[2000]), tf.float64)
    # data=features['feature']
    label = tf.cast(features['label'], tf.int32)
    return data, label

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 400000
    elif subset == 'eval':
      return 100000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)