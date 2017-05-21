"""This module contains methods that are not class dependent and useful in
   general settings."""

import tensorflow as tf


def get_trainables(session):
  """Get all the trainable variables in the current session."""
  result = {}
  for tvar in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    result[tvar.name] = session.run(tvar).tolist()
  return result
