"""This module contains methods that are not class dependent and useful in
   general settings."""

from abc import ABCMeta
import re

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


def split_train_test(pairs, ratio=.7):
  """Given a list of (input, label) pairs, return two separate lists, keeping
     `ratio` of the original data in the first returned list."""
  i = int(len(pairs) * ratio)
  return pairs[:i], pairs[i:]
