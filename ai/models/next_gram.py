"""TODO: add descriptive docstring."""

from __future__ import division

from six.moves import xrange
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.layers.core import Dense

from ai.models import BaseModel


class NextGram(BaseModel):
  """Language model architecture to predict the next n-gram."""
  
  def __init__(self, num_types=0, batch_size=20,
               num_steps=40, embedding_size=100, rnn_layers=2,
               max_grad_norm=5., rnn_cell=tf.contrib.rnn.LSTMBlockCell, **kw):
    """Keyword arguments:
       `num_types`: the vocabulary size (way faster to provide than infer),
       `lr`: the learning rate,
       `lr_decay`: learning rate decay (op must be manually called),
       `batch_size`: number of inputs per training step,
       `num_steps`: number of time steps for the RNN,
       `embedding_size`: dimensionality of the hidden units,
       `rnn_layers`: the number of RNN cells to be stacked,
       `max_grad_norm`: gradient norm clipping value to avoid exploding values,
       `rnn_cell`: the RNN class to be used (e.g. LSTM, GRU)."""
    self.num_types = num_types
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.embedding_size = embedding_size
    self.rnn_layers = rnn_layers
    self.max_grad_norm = max_grad_norm
    self.rnn_cell = rnn_cell
    super(NextGram, self).__init__(**kw)
  
  def build_graph(self):
    # Train/validation placeholders
    self.inputs = tf.placeholder(
      tf.int32, name='inputs', shape=[self.batch_size, self.num_steps])
    self.labels = tf.placeholder(
      tf.int32, name='labels', shape=[self.batch_size, self.num_steps])
    self.seed = tf.placeholder(tf.int32, name='seed', shape=[])
    self.temperature = tf.placeholder_with_default(
      1., name='temperature', shape=[])
    
    # Convert input id's to embeddings
    with tf.variable_scope('embeddings'):
      embedding_kernel = tf.get_variable(
        'kernel', [self.num_types, self.embedding_size],
        initializer=tf.random_uniform_initializer(minval=-1., maxval=1.)
      )
      embeddings = tf.nn.embedding_lookup(embedding_kernel, self.inputs)
    
    # Feed embeddings into RNN stack
    with tf.variable_scope('rnn_stack'):
      cell = tf.contrib.rnn.MultiRNNCell(
        [self.rnn_cell(self.embedding_size) for _ in xrange(self.rnn_layers)]
      )
      initial_state = cell.zero_state(self.batch_size, tf.float32)
      rnn_out, _ = tf.nn.dynamic_rnn(
        cell, embeddings, initial_state=initial_state
      )
      rnn_out = tf.reshape(
        tf.concat(axis=1, values=rnn_out), [-1, self.embedding_size]
      )
      dense = Dense(
        self.num_types, name='dense', activation=lambda x: x/self.temperature)
      logits = dense.apply(rnn_out)
      ### EXPERIMENTAL
      generative_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell, lambda x: tf.nn.embedding_lookup(embedding_kernel, x),
        tf.tile([self.seed], [1]), 1,
        initial_state, self.batch_size, output_layer=dense)
      tf.get_variable_scope().reuse_variables()
      self.generative_output = tf.contrib.seq2seq.dynamic_decode(
        generative_decoder, maximum_iterations=200)
    
    # Index outputs
    self.output = tf.argmax(logits, axis=1, name='output')
    
    # Softmax cross entropy and perplexity
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=tf.reshape(self.labels, [-1]), name='loss'
    ))
    with tf.name_scope('perplexity'):
      self.perplexity = tf.exp(loss)
      tf.summary.scalar('perplexity', self.perplexity)
    
    # Optimizer clips gradients by max norm
    with tf.variable_scope('train_op'):
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, tvars), self.max_grad_norm
      )
      optimizer = tf.train.AdamOptimizer(self.lr)
      self.train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step
      )
    
    # Operation for decaying learning rate
    with tf.name_scope('decay_lr'):
      self.decay_lr = tf.assign(self.lr, self.lr * self.lr_decay)
