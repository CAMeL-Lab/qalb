"""TODO: add descriptive docstring."""

from six.moves import xrange
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.layers.core import Dense

from ai.models import BaseModel


class Seq2Seq(BaseModel):
  """Sequence to Sequence model with an attention mechanism. Note that for the
     best results, the model assumes that the inputs and targets are
     preprocessed with the following conventions:
     1. All the inputs are padded with a unique `pad_id`,
     2. All the labels have a unique `eos_id` as the final token,
     3. A `go_id` is reserved for the model to provide to the decoder."""
  
  def __init__(self, num_types=0, max_encoder_length=99, max_decoder_length=99,
               pad_id=0, eos_id=1, go_id=2, batch_size=32, embedding_size=128,
               rnn_layers=2, bidirectional_encoder=True, max_grad_norm=5,
               use_lstm=False, use_residual=False, use_luong_attention=True,
               p_sample=0., **kw):
    """TODO: add documentation for all arguments."""
    self.num_types = num_types
    self.max_encoder_length = max_encoder_length
    self.max_decoder_length = max_decoder_length
    self.pad_id = pad_id
    self.eos_id = eos_id
    self.go_id = go_id
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.rnn_layers = rnn_layers
    self.bidirectional_encoder = bidirectional_encoder
    self.max_grad_norm = max_grad_norm
    self.use_lstm = use_lstm
    self.use_residual = use_residual
    self.use_luong_attention = use_luong_attention
    self.p_sample = tf.Variable(p_sample, trainable=False, name='p_sample')
    super(Seq2Seq, self).__init__(**kw)
  
  def build_graph(self):
    self.inputs = tf.placeholder(
      tf.int32, name='inputs',
      shape=[self.batch_size, self.max_encoder_length]
    )
    self.labels = tf.placeholder(
      tf.int32, name='labels',
      shape=[self.batch_size, self.max_decoder_length]
    )
    decoder_ids = tf.concat(
      [tf.tile([[self.go_id]], [self.batch_size, 1]), self.labels[:, 1:]], 1
    )
    # TODO: use a custom helper where we don't have to waste memory filling
    # all the timesteps with tokens that will never be used.
    decoder_seed_ids = tf.tile(
      [[self.go_id]], [self.batch_size, self.max_decoder_length]
    )
    
    with tf.variable_scope('embeddings'):
      sqrt3 = 3 ** .5  # Uniform(-sqrt3, sqrt3) has variance 1
      self.embedding_kernel = tf.get_variable(
        'kernel', [self.num_types, self.embedding_size],
        initializer=tf.random_uniform_initializer(minval=-sqrt3, maxval=sqrt3)
      )
      encoder_input = self.get_embeddings(self.inputs)
      decoder_input = self.get_embeddings(decoder_ids)
      decoder_seed = self.get_embeddings(decoder_seed_ids)
    
    with tf.variable_scope('encoder_rnn'):
      input_lengths = self.get_sequence_length(self.inputs)
      if self.bidirectional_encoder:
        (encoder_fw_out, encoder_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
          self.rnn_cell(), self.rnn_cell(), encoder_input, dtype=tf.float32,
          sequence_length=input_lengths
        )
        encoder_output = tf.layers.dense(
          tf.concat([encoder_fw_out, encoder_bw_out], 2), self.embedding_size,
          name='output_proj'
        )
      else:
        encoder_output, _ = tf.nn.dynamic_rnn(
          self.rnn_cell(), encoder_input, dtype=tf.float32,
          sequence_length=input_lengths
        )
      if self.rnn_layers > 1:
        encoder_cells = tf.contrib.rnn.MultiRNNCell(
          [self.rnn_cell() for _ in xrange(self.rnn_layers - 1)]
        )
        encoder_output, _ = tf.nn.dynamic_rnn(
          encoder_cells, encoder_output, dtype=tf.float32,
          sequence_length=input_lengths
        )
    
    with tf.variable_scope('decoder_rnn') as scope:
      # The first RNN is wrapped with the attention mechanism
      if self.use_luong_attention:
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
          self.embedding_size, encoder_output,
          memory_sequence_length=input_lengths
        )
      else:
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
          self.embedding_size, encoder_output,
          memory_sequence_length=input_lengths
        )
      decoder_cell = self.rnn_cell(attention_mechanism=attention_mechanism)
      # Stack all the cells if more than one RNN is used
      if self.rnn_layers > 1:
        decoder_cell = tf.contrib.rnn.MultiRNNCell(
          [decoder_cell] + [self.rnn_cell()
                            for _ in xrange(self.rnn_layers - 1)]
        )
      initial_state = decoder_cell.zero_state(self.batch_size, tf.float32)
      # Training decoder
      sampling_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
        decoder_input, tf.tile([self.max_decoder_length], [self.batch_size]),
        self.p_sample
      )
      decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, sampling_helper, initial_state
      )
      # Generative decoder
      generative_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
        decoder_seed, tf.tile([self.max_decoder_length], [self.batch_size]), 1.
      )
      generative_decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, generative_helper, initial_state
      )
      # No need to scope by using a layer instance. This can also be used in
      # decoder helpers that direcly output logits or their argmax.
      dense = Dense(self.num_types, name='dense')
      decoder_output = tf.contrib.seq2seq.dynamic_decode(decoder)
      logits = dense.apply(decoder_output[0].rnn_output)
      scope.reuse_variables()
      generative_output = tf.contrib.seq2seq.dynamic_decode(
        generative_decoder, maximum_iterations=self.max_decoder_length
      )
      generative_logits = dense.apply(generative_output[0].rnn_output)
    
    # Index outputs (greedy)
    self.output = tf.argmax(logits, axis=2, name='output')
    self.generative_output = tf.argmax(
      generative_logits, axis=2, name='generative_output'
    )
    
    # Weighted softmax cross entropy loss
    with tf.name_scope('loss'):
      mask = tf.cast(tf.sign(self.labels), tf.float32)
      self.loss = tf.contrib.seq2seq.sequence_loss(logits, self.labels, mask)
      tf.summary.scalar('log_perplexity', self.loss)
    
    # Optimizer clips gradients by max norm
    with tf.variable_scope('train_op'):
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(
        tf.gradients(self.loss, tvars), self.max_grad_norm
      )
      self.optimizer = tf.train.AdamOptimizer(self.lr)
      self.train_op = self.optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step
      )
  
  def get_embeddings(self, ids):
    """Performs embedding lookup. Useful as a method for decoder helpers."""
    return tf.nn.embedding_lookup(self.embedding_kernel, ids)
  
  def rnn_cell(self, attention_mechanism=None):
    """Get a new RNN cell with wrappers according to the initial config."""
    cell = None
    if self.use_lstm:
      cell = tf.contrib.rnn.LSTMBlockCell(self.embedding_size)
    else:
      cell = tf.contrib.rnn.GRUBlockCell(self.embedding_size)
    if attention_mechanism is not None:
      cell = tf.contrib.seq2seq.DynamicAttentionWrapper(
        cell, attention_mechanism, self.embedding_size
      )
    if self.use_residual:
      cell = tf.contrib.rnn.ResidualWrapper(cell)
    return cell
  
  def get_sequence_length(self, sequence_batch):
    """Given a 2D batch of input sequences, return a vector with the lengths
       of every sequence excluding the paddings."""
    with tf.name_scope('get_sequence_length'):
      return tf.reduce_sum(
        tf.sign(tf.abs(sequence_batch - self.pad_id)), reduction_indices=1
      )
