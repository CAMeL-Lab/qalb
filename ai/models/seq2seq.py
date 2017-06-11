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
               rnn_layers=2, bidirectional_encoder=True, add_fw_bw=True,
               pyramid_encoder=False, max_grad_norm=5., epsilon=1e-8,
               use_lstm=False, use_residual=False, use_luong_attention=True,
               beam_size=1, p_sample=0, feed_inputs_to_decoder=False, **kw):
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
    self.add_fw_bw = add_fw_bw
    self.pyramid_encoder = pyramid_encoder
    self.max_grad_norm = max_grad_norm
    self.epsilon = epsilon
    self.use_lstm = use_lstm
    self.use_residual = use_residual
    self.use_luong_attention = use_luong_attention
    self.beam_size = beam_size
    self.feed_inputs_to_decoder = feed_inputs_to_decoder
    self.p_sample = tf.Variable(
      p_sample, trainable=False, dtype=tf.float32, name='p_sample')
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
    
    with tf.name_scope('decoder_inputs'):
      if self.feed_inputs_to_decoder:
        decoder_ids = tf.concat(
          [tf.tile([[self.go_id]], [self.batch_size, 1]), self.inputs], 1
        )
      else:
        decoder_ids = tf.concat(
          [tf.tile([[self.go_id]], [self.batch_size, 1]), self.labels], 1
        )
    
    with tf.variable_scope('embeddings'):
      sqrt3 = 3 ** .5  # Uniform(-sqrt3, sqrt3) has variance 1
      self.embedding_kernel = tf.get_variable(
        'kernel', [self.num_types, self.embedding_size],
        initializer=tf.random_uniform_initializer(minval=-sqrt3, maxval=sqrt3)
      )
      encoder_input = self.get_embeddings(self.inputs)
      decoder_input = self.get_embeddings(decoder_ids)
    
    with tf.variable_scope('encoder'):
      encoder_output = self.build_encoder(encoder_input)
    
    with tf.variable_scope('decoder'):
      logits, self.generative_output = self.build_decoder(
        encoder_output, decoder_input
      )
    
    # Index outputs (greedy)
    self.output = tf.argmax(logits, axis=2, name='output')
    
    # Weighted softmax cross entropy loss
    with tf.name_scope('loss'):
      mask = tf.cast(tf.sign(self.labels), tf.float32)
      loss = tf.contrib.seq2seq.sequence_loss(logits, self.labels, mask)
      self.perplexity = tf.exp(loss)
    tf.summary.scalar('perplexity', self.perplexity)
    
    # Optimizer clips gradients by max norm
    with tf.variable_scope('train_op'):
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, tvars), self.max_grad_norm
      )
      self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=self.epsilon)
      self.train_op = self.optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step
      )
  
  
  def build_encoder(self, encoder_input):
    input_lengths = self.get_sequence_length(self.inputs)
    if self.bidirectional_encoder:
      (encoder_fw_out, encoder_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
        self.rnn_cell(), self.rnn_cell(), encoder_input, dtype=tf.float32,
        sequence_length=input_lengths
      )
      if self.add_fw_bw:
        encoder_output = encoder_fw_out + encoder_bw_out
      else:
        encoder_output = tf.layers.dense(
          tf.concat([encoder_fw_out, encoder_bw_out], 2), self.embedding_size,
          activation=tf.tanh, name='bidirectional_projection'
        )
    else:
      encoder_output, _ = tf.nn.dynamic_rnn(
        self.rnn_cell(), encoder_input, dtype=tf.float32,
        sequence_length=input_lengths
      )
    if self.rnn_layers > 1:
      if self.pyramid_encoder:
        for i in xrange(1, self.rnn_layers):
          # Concatenate adjacent pairs and reshape them to their original size
          eo_sh = encoder_output.get_shape()
          concat_shape = map(int, [eo_sh[0], eo_sh[1] / 2, eo_sh[2] * 2])
          encoder_output = tf.layers.dense(
            tf.reshape(encoder_output, concat_shape), self.embedding_size,
            name='pyramid_projection_{}'.format(i)
          )
          encoder_output, _ = tf.nn.dynamic_rnn(
            self.rnn_cell(self.embedding_size / (2 ** i)),
            tf.reshape(encoder_output, concat_shape), dtype=tf.float32
          )
      else:
        encoder_cells = tf.contrib.rnn.MultiRNNCell(
          [self.rnn_cell() for _ in xrange(self.rnn_layers - 1)]
        )
        encoder_output, _ = tf.nn.dynamic_rnn(
          encoder_cells, encoder_output, dtype=tf.float32,
          sequence_length=input_lengths
        )
    return tf.contrib.seq2seq.tile_batch(encoder_output, self.beam_size)
  
  
  def build_decoder(self, encoder_output, decoder_input):
    # The first RNN is wrapped with the attention mechanism
    if self.use_luong_attention:
      attention_mechanism = tf.contrib.seq2seq.LuongAttention
    else:
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention
    decoder_cell = self.rnn_cell(attention_mechanism=attention_mechanism(
      self.embedding_size, encoder_output
    ))
    # Use last state of encoder to avoid wrong first outputs
    initial_state_pass = tf.split(tf.layers.dense(
      encoder_output[:, 0], self.embedding_size * self.rnn_layers,
      activation=tf.tanh, name='initial_decoder_state'
    ), self.rnn_layers, axis=1)
    beam_batch_size = self.batch_size * self.beam_size
    initial_state = tf.contrib.seq2seq.AttentionWrapperState(
      cell_state=initial_state_pass[0],
      attention=tf.zeros([beam_batch_size, self.embedding_size]),
      alignments=tf.zeros([beam_batch_size, self.max_decoder_length]),
      time=tf.zeros(()), alignment_history=()
    )
    # Stack all the cells if more than one RNN is used
    if self.rnn_layers > 1:
      decoder_cell = tf.contrib.rnn.MultiRNNCell(
        [decoder_cell] + [self.rnn_cell()
                          for _ in xrange(self.rnn_layers - 1)]
      )
      initial_state = tuple([initial_state] + list(initial_state_pass[1:]))
    # Training decoder
    sampling_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
      tf.contrib.seq2seq.tile_batch(decoder_input, self.beam_size),
      tf.tile([self.max_decoder_length], [beam_batch_size]),
      self.p_sample
    )
    decoder = tf.contrib.seq2seq.BasicDecoder(
      decoder_cell, sampling_helper, initial_state
    )
    dense = Dense(self.num_types, name='dense')
    decoder_output = tf.contrib.seq2seq.dynamic_decode(decoder)
    logits = dense.apply(decoder_output[0].rnn_output)
    # TODO: allow custom length penalty weight
    generative_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
      decoder_cell, self.get_embeddings,
      tf.tile([self.go_id], [self.batch_size]), self.eos_id,
      initial_state, self.beam_size, output_layer=dense,
      length_penalty_weight=.6
    )
    tf.get_variable_scope().reuse_variables()
    generative_output = tf.contrib.seq2seq.dynamic_decode(
      generative_decoder, maximum_iterations=self.max_decoder_length
    )
    return logits, generative_output[0].beam_search_decoder_output
  
  
  def get_embeddings(self, ids):
    """Performs embedding lookup. Useful as a method for decoder helpers."""
    return tf.nn.embedding_lookup(self.embedding_kernel, ids)
  
  
  def rnn_cell(self, num_units=None, attention_mechanism=None):
    """Get a new RNN cell with wrappers according to the initial config."""
    cell = None
    if num_units is None:
      num_units = self.embedding_size
    if self.use_lstm:
      cell = tf.contrib.rnn.LSTMBlockCell(num_units)
    else:
      cell = tf.contrib.rnn.GRUBlockCell(num_units)
    if attention_mechanism is not None:
      cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism)
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
