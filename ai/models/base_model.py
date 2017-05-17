from __future__ import print_function

import os

import tensorflow as tf


class BaseModel(object):
  """Model abstraction to handle restoring variables and summary writers."""
  
  def __init__(self, restore=False, model_name='default', verbose=True):
    """Builds the model and the computational graph.
       Keyword arguments:
       `restore`: whether to override or use the model's pre-trained variables,
       `model_name`: string for the model's checkpoint and summary outputs,
       `verbose`: whether to print logging messages."""
    if verbose: print("Building computational graph...")
    self.build_graph()
    self.summary_op = tf.summary.merge_all()
    # Default configurations
    self.restore = restore
    self.model_name = model_name
    self.verbose = verbose
  
  def build_graph(self):
    """Override. All placeholders and operations that define the computational
       graph should be defined in this method."""
    pass
  
  def start(self):
    """Initialize or restore the model, and create the summary writer. This
       method expects to be called within a session whose graph is that where
       the `build_graph` method was previously called."""
    
    sess = tf.get_default_session()
    output_dir = os.path.join('output', self.model_name)
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    
    # Check whether to restore the model
    self.saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(output_dir)
    if self.restore and ckpt:
      if self.verbose: print("Restoring model...")
      self.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      if self.verbose: print("Initializing model...")
      sess.run(tf.global_variables_initializer())
      # Flush the provided summary writer to avoid overwriting
      if tf.gfile.Exists(output_dir):
        tf.gfile.DeleteRecursively(output_dir)
      tf.gfile.MakeDirs(output_dir)
      tf.gfile.MakeDirs(train_dir)
      tf.gfile.MakeDirs(valid_dir)
    
    # Create the summary writers
    self.train_writer = tf.summary.FileWriter(train_dir, graph=sess.graph)
    self.valid_writer = tf.summary.FileWriter(valid_dir, graph=sess.graph)
  
  def save(self, filename='checkpoint.ckpt'):
    """Save the model's variables for later training or evaluation. Requires
       the model's graph to be fully built."""
    sess = tf.get_default_session()
    self.saver.save(sess, os.path.join('output', self.model_name, filename), global_step=self.global_step.eval())
  
  def export(self):
    """Save all the variables of the model for non-python replications."""
    result = {}
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
      result[v.name[:-2]] = v.eval().tolist()
    return result
