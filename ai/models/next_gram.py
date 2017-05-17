from six.moves import xrange

import os

import tensorflow as tf

from base_models import Model


class NextGram(Model):
  """Language model to predict the next n-gram."""
  
  def __init__(self, alphabet_size=0, lr=1., lr_decay=1., batch_size=20,
    num_steps=40, embed_size=100, num_rnn_layers=2, max_grad_norm=5.,
    rnn_cell=tf.contrib.rnn.LSTMCell, **kw):
    self.lr = lr
    self.lr_decay = lr_decay
    self.alphabet_size = alphabet_size
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.embed_size = embed_size
    self.num_rnn_layers = num_rnn_layers
    self.max_grad_norm = max_grad_norm
    self.rnn_cell = rnn_cell
    super(NextGram, self).__init__(**kw)
  
  def build_graph(self):
    # Train/validation placeholders
    self.inputs = tf.placeholder(tf.int32, name='inputs',
      shape=[self.batch_size, self.num_steps])
    self.labels = tf.placeholder(tf.int32, name='labels',
      shape=[self.batch_size, self.num_steps])
    
    # Variables that the model can modify during training
    self._lr = tf.Variable(self.lr, trainable=False, name='lr')
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # Convert input id's to embeddings
    with tf.variable_scope('embeddings'):
      E = tf.get_variable('kernel', [self.alphabet_size, self.embed_size],
        initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
      embeds = tf.nn.embedding_lookup(E, self.inputs)
      self.get_embeddings = tf.nn.embedding_lookup(E, self.tokens)
    
    # Feed embeddings into LSTM stack
    with tf.variable_scope('lstm_stack') as scope:
      ### Train/validation feed
      cell = tf.contrib.rnn.MultiRNNCell(
        [self.rnn_cell(self.embed_size) for _ in xrange(self.num_rnn_layers)])
      self.initial_state = cell.zero_state(self.batch_size, tf.float32)
      rnn_out, _ = tf.nn.dynamic_rnn(cell, embeds,
        initial_state=self.initial_state)
      rnn_out = tf.reshape(tf.concat(axis=1, values=rnn_out),
        [-1, self.embed_size])
    
    # Softmax layer
    logits = tf.layers.dense(rnn_out, self.alphabet_size, name='dense')
    gen_logits = tf.layers.dense(gen_rnn_out, self.alphabet_size, reuse=True,
      name='dense')
    
    # Index outputs
    with tf.name_scope('output'):
      self.output = tf.argmax(logits, axis=1)
      self.gen_output = (tf.argmax(gen_logits, axis=1), gen_state)
    
    # Softmax cross entropy and perplexity
    with tf.name_scope('perplexity'):
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.reshape(self.labels, [-1])))
      self.perplexity = tf.exp(loss)
      tf.summary.scalar('perplexity', self.perplexity)
    
    # Optimizer clips gradients by max norm
    with tf.variable_scope('train_op'):
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
        self.max_grad_norm)
      optimizer = tf.train.AdamOptimizer(self._lr, epsilon=1e-2)
      self.train_op = optimizer.apply_gradients(zip(grads, tvars),
        global_step=self.global_step)
    
    # Ops for decaying learning rate
    with tf.name_scope('decay_lr'):
      self.decay_lr = tf.assign(self._lr, self._lr * self.lr_decay)
