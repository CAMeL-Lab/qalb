"""This module contains methods that are not class dependent and useful in
   general settings."""

from abc import ABCMeta
import re

import tensorflow as tf
from tensorflow.python.client import device_lib


### GENERAL

def abstractclass(cls):
  """Abstract class decorator with compatibility for python 2 and 3."""
  orig_vars = cls.__dict__.copy()
  slots = orig_vars.get('__slots__')
  if slots is not None:
    if isinstance(slots, str):
      slots = [slots]
    for slots_var in slots:
      orig_vars.pop(slots_var)
  orig_vars.pop('__dict__', None)
  orig_vars.pop('__weakref__', None)
  return ABCMeta(cls.__name__, cls.__bases__, orig_vars)


### DATASET UTILS

def split_train_test(pairs, ratio=.7):
  """Given a list of (input, label) pairs, return two separate lists, keeping
     `ratio` of the original data in the first returned list."""
  i = int(len(pairs) * ratio)
  return pairs[:i], pairs[i:]


### GRAPH UTILS

def dense_to_sparse(t, shape=None, name='to_sparse'):
  """Givn a dense tensor `t`, return its `SparseTensor` equivalent."""
  with tf.name_scope(name):
    indices = tf.where(tf.not_equal(t, 0))
    if shape is None:
      shape = t.get_shape()
    return tf.SparseTensor(
      indices=indices, values=tf.gather_nd(t, indices), dense_shape=shape)

def edit_distance(t1, t2, shapes=None, name='edit_distance'):
  """Return the average edit distance between `t1` and `t2`."""
  with tf.name_scope(name):
    if shapes is None:
      shapes = [t1.get_shape(), t2.get_shape()]
    tf.reduce_mean(tf.edit_distance(dense_to_sparse(t1), dense_to_sparse(t2)))


### SESSION UTILS

def get_trainables():
  """Get all the trainable variables in the current default session."""
  sess = tf.get_default_session()
  result = {}
  for tvar in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    result[tvar.name] = sess.run(tvar).tolist()
  return result


### HARDWARE UTILS

def get_available_gpus():
  """Get all available GPUs in a list."""
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']


### MISC UTILS

def max_repetitions(s, threshold=8):
  """Find the largest contiguous repeating substring in s, repeating itself at
     least `threshold` times. Example:
     >>> max_repetitions("blablasbla")  # returns ['bla', 2]."""
  repetitions_re = re.compile(r'(.+?)\1{%d,}' % threshold)
  max_repeated = None
  for match in repetitions_re.finditer(s):
    new_repeated = [match.group(1), len(match.group(0))/len(match.group(1))]
    # pylint: disable=unsubscriptable-object
    if max_repeated is None or max_repeated[1] < new_repeated[1]:
      max_repeated = new_repeated
  return max_repeated
