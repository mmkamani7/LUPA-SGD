"""Implementation of distributed training using Local Updates with Periodic Averaging (LUPA) SGD
Haddadpour, F.,  Kamani, M.M., Mahdavi, M., & Cadambe, V.
"Local SGD with Periodic Averaging: Tighter Analysis and Adaptive Synchronization"
 Advances in Neural Information Processing. 2019.

Support single-host training with one or multiple devices.
"""
from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import os
import json
from collections import namedtuple

import utils
import dataloader as dl
import numpy as np
import six
from six.moves import xrange 
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

tf.logging.set_verbosity(tf.logging.INFO)

"""
__author__ = "Mohammad Mahdi Kamani"
__copyright__ = "Copyright 2019, Mohammad Mahdi Kamani"

__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Madhi Kamani"
__status__ = "Prototype"
"""

def get_model_fn(num_gpus, variable_strategy, num_workers, run_type='local'):
  """Returns a function that will build the resnet model."""

  def _linearregression_model_fn_sync(features, labels, mode, params):
    """Resnet model body.

    Support single host, one or more GPU training. Parameter distribution can
    be either one of the following scheme.
    1. CPU is the parameter server and manages gradient updates.
    2. Parameters are distributed evenly across all GPUs, and the first GPU
       manages gradient updates.

    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    weight_decay = params.weight_decay


    features = features[0:num_gpus]
    labels = labels[0:num_gpus]
    tower_features = features
    tower_labels = labels
    tower_losses = []
    tower_gradvars = []
    tower_preds = []


    if num_gpus == 0:
      num_devices = 1
      device_type = 'cpu'
    else:
      num_devices = num_gpus
      device_type = 'gpu'

    for i in range(num_devices):
      worker_device = '/{}:{}'.format(device_type, i)
      if variable_strategy == 'CPU':
        device_setter = utils.local_device_setter(
            worker_device=worker_device)
      elif variable_strategy == 'GPU':
        device_setter = utils.local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
      with tf.variable_scope('LinearRegression', reuse=bool(i != 0)) as var_scope:
        with tf.name_scope('tower_%d' % i) as name_scope:
          with tf.device(device_setter):
            loss, gradvars, preds = _tower_fn(
                is_training, weight_decay, tower_features[i], tower_labels[i],
                params.feature_dim, var_scope.name, params.problem)
            tower_losses.append(loss)
            tower_gradvars.append(gradvars)
            tower_preds.append(preds)

    # Now compute global loss and gradients.
    gradvars = []
    with tf.name_scope('gradient_averaging'):
      all_grads = {}
      for grad, var in itertools.chain(*tower_gradvars):
        if grad is not None:
          all_grads.setdefault(var, []).append(grad)
      for var, grads in six.iteritems(all_grads):
        # Average gradients on the same device as the variables
        # to which they apply.
        with tf.device(var.device):
          if len(grads) == 1:
            avg_grad = grads[0]
          else:
            avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
        gradvars.append((avg_grad, var))

    # Device that runs the ops to apply global gradient updates.
    consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
    with tf.device(consolidation_device):
      loss = tf.reduce_mean(tower_losses, name='loss')

      examples_sec_hook = utils.ExamplesPerSecondHook(
          params.train_batch_size, every_n_steps=100)

      tensors_to_log = {'loss': loss}

      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)

      train_hooks = [logging_hook, examples_sec_hook]

      # optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.learning_rate)
      optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

      if params.run_type == 'sync':
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer, replicas_to_aggregate=num_workers)
        sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
        train_hooks.append(sync_replicas_hook)

      # Create single grouped train op
      train_op = [
          optimizer.apply_gradients(
              gradvars, global_step=tf.train.get_global_step())
      ]

      train_op = tf.group(*train_op)


    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=train_hooks)

  def _linearregression_model_fn_local(features, labels, mode, params):
    """

    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    weight_decay = params.weight_decay

    # features = features[0:num_gpus]
    # labels = labels[0:num_gpus]
    tower_features = features
    tower_labels = labels
    tower_losses = []
    tower_ops= []
    tower_preds = []
    var_scopes=[]


    if num_gpus == 0:
      num_devices = 1
      device_type = 'cpu'
    else:
      num_devices = num_gpus
      device_type = 'gpu'

    for i in range(num_devices):
      worker_device = '/{}:{}'.format(device_type, i)
      if variable_strategy == 'CPU':
        device_setter = utils.local_device_setter(
            worker_device=worker_device)
        # device_setter = tf.train.replica_device_setter(
        #     worker_device=worker_device)
      elif variable_strategy == 'GPU':
        device_setter = utils.local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
        # device_setter = tf.train.replica_device_setter(
        #     ps_device=worker_device,
        #     worker_device=worker_device
        # )
      with tf.variable_scope('LinearRegression_{}'.format(i)) as var_scope:
        with tf.name_scope('tower_%d' % i) as name_scope:
          with tf.device(device_setter):
            loss, gradvars, preds = _tower_fn(
                is_training, weight_decay, tower_features[i], tower_labels[i],
                params.feature_dim, var_scope.name, params.problem)
            var_scopes.append(var_scope.name)
            
            tower_losses.append(loss)
            # tower_gradvars.append(gradvars)
            tower_preds.append(preds)

            global_step = tf.cast(tf.train.get_global_step(), tf.float32)
            lr = params.learning_rate
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=params.learning_rate,momentum=0.97)

            # Create single grouped train op
            train_op = [
                optimizer.apply_gradients(
                    gradvars, global_step=tf.train.get_global_step(), name='apply_gradient_tower_{}'.format(i))
            ]
            tower_ops.append(train_op)


    # Device that runs the ops to apply global gradient updates.
    consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
    with tf.device(consolidation_device):

      examples_sec_hook = utils.ExamplesPerSecondHook(
        params.train_batch_size * (1 + params.redundancy), every_n_steps=100)
      loss = tf.reduce_mean(tower_losses, name='loss')
      tensors_to_log = {'loss': loss}
      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)
      train_hooks = [ logging_hook, examples_sec_hook]
      if params.run_type == 'multi':
        if params.adaptive:
          alpha=2/(params.num_comm+1) * (params.train_steps/(params.num_comm * params.sync_step))
          local_updates = [params.sync_step * (1+alpha * i) for i in range(params.num_comm+1)]
          sync_hook = utils.SyncHook(scopes=var_scopes, every_n_steps=params.sync_step, adaptive=local_updates)
        else:
          sync_hook = utils.SyncHook(scopes=var_scopes, every_n_steps=params.sync_step)
        train_hooks.append(sync_hook)


      train_ops = tf.group(*tower_ops)


    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_ops,
        training_hooks=train_hooks
        )


  if run_type in ['sync', 'async']:
    return _linearregression_model_fn_sync
  else:
    return _linearregression_model_fn_local



def _tower_fn(is_training, weight_decay, feature, label,
              feature_dim, scope, problem):
  """Build computation tower for Linear Regression.

  Args:
    is_training: true if is training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    data_format: channels_last (NHWC) or channels_first (NCHW).
    num_layers: number of layers, an int.
    batch_norm_decay: decay for batch normalization, a float.
    batch_norm_epsilon: epsilon for batch normalization, a float.
    scope: is the scope name that this tower is building its graph on
    dataset_name: choices between cifar10 and cifar100

  Returns:
    A tuple with the loss for the tower, the gradients and parameters, and
    predictions.

  """

  if problem == 'linear':
    W = tf.get_variable('weight',shape=[feature_dim,1], dtype =tf.float32)
    b = tf.get_variable('bias', shape=[1], dtype=tf.float32)
    tower_pred = tf.matmul(feature,W) + b
    tower_pred = tf.squeeze(tower_pred)
    tower_loss = tf.losses.mean_squared_error(label,tower_pred) + weight_decay * tf.nn.l2_loss(W)

  elif problem == 'mnist':
    layer = tf.layers.Dense(units=10)
    logits = layer(feature)
    tower_pred = tf.nn.softmax(logits)
    tower_loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=logits) +  weight_decay * tf.add_n(
																			[tf.nn.l2_loss(v) for v in layer.weights])
                              
  elif problem == 'adult':
    layer = tf.layers.Dense(units=2)
    logits = layer(feature)
    tower_pred = tf.nn.softmax(logits)
    model_params = tf.trainable_variables(scope=scope)
    tower_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=label)
    tower_loss +=  weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in model_params])

  elif problem == 'epsilon':
    layer = tf.layers.Dense(units=2)
    logits = layer(feature)
    tower_pred = tf.nn.softmax(logits)
    model_params = tf.trainable_variables(scope=scope)
    tower_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=label)

  
  model_params = tf.trainable_variables(scope=scope)
  tower_grad = tf.gradients(tower_loss, model_params)

  return tower_loss, zip(tower_grad, model_params), tower_pred


def input_fn(data_dir,
             subset,
             num_shards,
             batch_size,
             problem='epsilon'):
    if problem == 'mnist':
        dataset = dl.MnistDataset(data_dir=data_dir,
                                   subset=subset,
                                   num_shards=num_shards)
    elif problem == 'adult':
        dataset = dl.AdultDataset(data_dir=data_dir,
                                   subset=subset,
                                   num_shards=num_shards)
    elif problem == 'epsilon':
        dataset = dl.EpsilonDataset(data_dir=data_dir,
                                   subset=subset,
                                   num_shards=num_shards)
    else:
        raise ValueError('The problem {} is not defined yet!'.format(problem))
    feature_shards, label_shards = dataset.make_batch(batch_size)
    return feature_shards, label_shards


def main(job_dir, num_gpus, variable_strategy, log_device_placement, num_intra_threads, **hparams):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  if hparams['config_path']:
      TF_CONFIG = json.load(open(hparams['config_path'], "r"))
      TF_CONFIG['model_dir'] = job_dir
      os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)

  # Session configuration.
  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=log_device_placement,
      intra_op_parallelism_threads=num_intra_threads,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True))

  config = utils.RunConfig(
      session_config=sess_config, model_dir=job_dir)
  if hparams['eval']:
    config = config.replace(save_checkpoints_steps=500)


  train_input_fn = functools.partial(
      input_fn,
      data_dir=hparams['data_dir'],
      subset='train',
      num_shards=num_gpus,
      batch_size=hparams['train_batch_size'],
      problem=hparams['problem'])

  eval_input_fn = functools.partial(
      input_fn,
      data_dir=hparams['data_dir'],
      subset='test',
      num_shards=num_gpus,
      batch_size=hparams['eval_batch_size'],
      problem=hparams['problem'])

  if hparams['eval_size'] % hparams['eval_batch_size'] != 0:
      raise ValueError(
          'validation set size must be multiple of eval_batch_size')

  train_steps = hparams['train_steps']
  eval_steps = hparams['eval_size'] // hparams['eval_batch_size']

  train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps, start_delay_secs=0, throttle_secs=30)

  classifier = tf.estimator.Estimator(
      model_fn=get_model_fn(num_gpus, variable_strategy,
                            config.num_worker_replicas or 1,
                            run_type=hparams['run_type']),
      config=config,
      params=tf.contrib.training.HParams(
                is_chief=config.is_chief,
                **hparams)
  )

  # Create experiment.
  tf.estimator.train_and_evaluate(
      estimator=classifier,
      train_spec=train_spec,
      eval_spec=eval_spec)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      required=True,
      help='The directory where the data is stored.')
  parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='The directory where the model will be stored.')
  parser.add_argument(
      '--variable-strategy',
      choices=['CPU', 'GPU'],
      type=str,
      default='CPU',
      help='Where to locate variable operations')
  parser.add_argument(
      '--num-gpus',
      type=int,
      default=1,
      help='The number of gpus used. Uses only CPU if set to 0.')
  parser.add_argument(
      '--train-steps',
      type=int,
      default=80000,
      help='The number of steps to use for training.')
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=128,
      help='Batch size for training.')
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=100,
      help='Batch size for validation.')

  parser.add_argument(
      '--weight-decay',
      type=float,
      default=2e-4,
      help='Weight decay for convolutions.')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.1,
      help="""\
      This is the inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """)

  parser.add_argument(
      '--num-intra-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for intra-op parallelism. When training on CPU
      set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.\
      """)
  parser.add_argument(
      '--num-inter-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """)
  parser.add_argument(
      '--log-device-placement',
      action='store_true',
      default=True,
      help='Whether to log device placement.')
  parser.add_argument(
      '--sync-step',
      type=int,
      default=100,
      help='Sync step for local version')
  parser.add_argument(
      '--run-type',
      type=str,
      default='local',
      choices=['sync','async','local','multi'],
      help='The type for running the experiment')
  parser.add_argument(
      '--config-path',
      type=str,
      default=None,
      help='The path to json file of config')
  parser.add_argument(
      '--eval',
      action='store_true',
      default=False,
      help="""If present when running in a distributed environment will run on eval mode.""")
  parser.add_argument(
      '--adaptive',
      action='store_true',
      default=False,
      help="""If present running adaptive local updates""")
  parser.add_argument(
      '--num-comm',
      type=int,
      default=100,
      help='Number of communication for adaptive local version')
  parser.add_argument(
      '--problem',
      type=str,
      default='linear',
      choices=['mnist','adult', 'epsilon']
  )
  args = parser.parse_args()

  if args.num_gpus > 0:
    assert tf.test.is_gpu_available(), "Requested GPUs but none found."
  if args.num_gpus < 0:
    raise ValueError(
        'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
  if args.num_gpus == 0 and args.variable_strategy == 'GPU':
    raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                     '--variable-strategy=CPU.')

  args.redundancy *= args.num_gpus
  main(**vars(args))
