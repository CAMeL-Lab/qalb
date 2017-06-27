"""This module contains methods that are not class dependent and useful in
   general settings."""

from abc import ABCMeta

import tensorflow as tf


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


def get_trainables():
  """Get all the trainable variables in the current default session."""
  sess = tf.get_default_session()
  result = {}
  for tvar in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    result[tvar.name] = sess.run(tvar).tolist()
  return result


def split_train_test(pairs, ratio=.7):
  """Given a list of (input, label) pairs, return two separate lists, keeping
     `ratio` of the original data in the first returned list."""
  i = int(len(pairs) * ratio)
  return pairs[:i], pairs[i:]
