"""TODO: add descriptive docstring."""

from __future__ import division

import os

from six.moves import xrange
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.init_ops import Initializer

from ai.models import BaseModel


class Seq2Seq(BaseModel):
  """Sequence to Sequence model with an attention mechanism. Note that for the
     best results, the model assumes that the inputs and targets are
     preprocessed with the following conventions:
     1. All the inputs are padded with a unique `pad_id`,
     2. All the labels have a unique `eos_id` as the final token,
     3. A `go_id` is reserved for the model to provide to the decoder."""
  
  def __init__(self, num_types=0, max_encoder_length=99, max_decoder_length=99,
               pad_id=0, eos_id=1, go_id=2,
               batch_size=32, embedding_size=32, hidden_size=256, rnn_layers=2,
               train_embeddings=True, default_embedding_matrix=None,
               bidirectional_encoder=False, bidirectional_mode='add',
               use_lstm=False, use_residual=False, attention=None,
               feed_inputs=False, dropout=1.,
               max_grad_norm=5., epsilon=1e-8, beam_size=1, **kw):
    """Keyword args:
       `num_types`: number of unique types (e.g. vocabulary or alphabet size),
       `max_encoder_length`: max length of the encoder,
       `max_decoder_length`: max length of the decoder,
       `pad_id`: the integer id that represents padding (defaults to 0),
       `eos_id`: the integer id that represents the end of the sequence,
       `go_id`: the integer id fed to the decoder as the first input,
       `batch_size`: minibatch size,
       `embedding_size`: dimensionality of the embeddings,
       `hidden_size`: dimensionality of the hidden units for the RNNs,
       `rnn_layers`: number of RNN layers for the encoder and decoder,
       `train_embeddings`: whether to do backprop on the embeddings,
       `default_embedding_matrix`: if None, set to a random uniform
        distribution with mean 0 and variance 1,
       `bidirectional_encoder`: whether to use a bidirectional encoder RNN,
       `bidirectional_mode`: string for the bidirectional RNN architecture:
        'add' (default): add the forward and backward hidden states,
        'project': use a projection matrix to resize the concatenation of the
                   forward and backward hidden states to `embedding_size`,
        'concat': concatenate the forward and backward inputs and pass that
                  as the input to the next RNN (note: this will not allow the
                  use of residual connections),
       `use_lstm`: set to False to use a GRU cell (Cho et al.,
        https://arxiv.org/abs/1406.1078),
       `use_residual`: whether to use residual connections between RNN cells
        (Wu et al., https://arxiv.org/pdf/1609.08144.pdf),
       `attention`: 'bahdanau', or 'luong' (none by default),
       
       # TODO: implement this feature
       `feed_inputs`: set to True to feed attention-based inputs to the
        decoder RNN (Luong et al., https://arxiv.org/abs/1508.04025),
        
       `dropout`: keep probability for the non-recurrent connections between
        RNN cells. Defaults to 1.0; i.e. no dropout,
       `max_grad_norm`: clip gradients to maximally this norm,
       `epsilon`: small numerical constant for AdamOptimizer (default 1e-8)."""
    self.num_types = num_types
    self.max_encoder_length = max_encoder_length
    self.max_decoder_length = max_decoder_length
    self.pad_id = pad_id
    self.eos_id = eos_id
    self.go_id = go_id
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.train_embeddings = train_embeddings
    self.default_embedding_matrix = default_embedding_matrix
    self.rnn_layers = rnn_layers
    self.bidirectional_encoder = bidirectional_encoder
    self.bidirectional_mode = bidirectional_mode
    self.use_lstm = use_lstm
    self.use_residual = use_residual
    self.attention = attention
    self.feed_inputs = feed_inputs
    self.dropout = dropout
    self.max_grad_norm = max_grad_norm
    self.epsilon = epsilon
    self.beam_size = beam_size
    # Use graph variables for learning rates to allow them to be modified/saved
    self.lr = tf.Variable(
      1e-3, trainable=False, dtype=tf.float32, name='learning_rate')
    # Sampling probability variable that can be manually changed. See scheduled
    # sampling paper (Bengio et al., https://arxiv.org/abs/1506.03099)
    self.p_sample = tf.Variable(
      0., trainable=False, dtype=tf.float32, name='sampling_probability')
    super(Seq2Seq, self).__init__(**kw)
  
  
  def start(self):
    super(Seq2Seq, self).start()
    # Summary writer for generative output
    infer_dir = os.path.join('output', self.model_name, 'infer')
    if not self.restore:
      tf.gfile.MakeDirs(infer_dir)
    sess = tf.get_default_session()
    self.infer_writer = tf.summary.FileWriter(infer_dir, graph=sess.graph)
  
  
  def build_graph(self):
    
    # Placeholders
    self.inputs = tf.placeholder(
      tf.int32, name='inputs',
      shape=[self.batch_size, self.max_encoder_length])
    self.labels = tf.placeholder(
      tf.int32, name='labels',
      shape=[self.batch_size, self.max_decoder_length])
    self.temperature = tf.placeholder_with_default(
      1., name='temperature', shape=[])
    
    # Sequence lengths - used throughout model
    with tf.name_scope('input_lengths'):
      self.input_lengths = tf.reduce_sum(
        tf.sign(tf.abs(self.inputs - self.pad_id)), reduction_indices=1)
    
    # Embedding matrix
    with tf.variable_scope('embeddings'):
      if self.default_embedding_matrix is not None:
        if isinstance(self.default_embedding_matrix, Initializer):
          initializer = self.default_embedding_matrix
        else:
          initializer = tf.constant_initializer(self.default_embedding_matrix)
      else:
        sq3 = 3 ** .5  # Uniform(-sqrt3, sqrt3) has variance 1
        # pylint: disable=redefined-variable-type
        initializer = tf.random_uniform_initializer(minval=-sq3, maxval=sq3)
      
      self.embedding_kernel = tf.get_variable(
        'kernel', [self.num_types, self.embedding_size],
        trainable=self.train_embeddings, initializer=initializer)
    
    # Look up the embeddings for the encoder and decoder inputs
    encoder_input = self.get_embeddings(self.inputs)
    
    with tf.variable_scope('encoder'):
      encoder_output = self.build_encoder(encoder_input)
    
    with tf.variable_scope('decoder'):
      logits, self.generative_output = self.build_decoder(encoder_output)
    
    # Softmax cross entropy loss masked by the target sequence lengths
    with tf.name_scope('loss'):
      mask = tf.cast(tf.sign(self.labels), tf.float32)
      loss = tf.contrib.seq2seq.sequence_loss(logits, self.labels, mask)
    
    self.perplexity = tf.exp(loss, name='perplexity')
    self.perplexity_summary = tf.summary.scalar('perplexity', self.perplexity)
    
    # Index outputs (greedy)
    self.output = tf.argmax(
      logits, axis=2, name='output', output_type=tf.int32)
    
    # Compute the edit distance for evaluations
    with tf.name_scope('edit_distance'):
      tensor_shape = [self.batch_size, self.max_decoder_length]
      self.lev_in = tf.placeholder(tf.int32, name='lev_in', shape=tensor_shape)
      # Sparse tensor for inputs
      lev_in_indices = tf.where(tf.not_equal(self.lev_in, 0))
      hypothesis = tf.SparseTensor(
        indices=lev_in_indices,
        values=tf.gather_nd(self.lev_in, lev_in_indices),
        dense_shape=tensor_shape)
      # Sparse tensor for labels
      labels_indices = tf.where(tf.not_equal(self.lev_in, 0))
      truth = tf.SparseTensor(
        indices=labels_indices,
        values=tf.gather_nd(self.labels, labels_indices),
        dense_shape=tensor_shape)
      self.lev_out = tf.reduce_mean(tf.edit_distance(hypothesis, truth))
    self.lev_summary = tf.summary.scalar('edit_distance', self.lev_out)
    
    # Adam and gradient descent optimizers with norm clipping. This prevents
    # exploding gradients and allows a switch from Adam to SGD when the model
    # is reaching convergence (Wu et al., https://arxiv.org/pdf/1609.08144.pdf)
    with tf.name_scope('train_ops'):
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, tvars), self.max_grad_norm)
      # Adam optimizer op (first and second order momenta)
      adam_optimizer = tf.train.AdamOptimizer(self.lr, epsilon=self.epsilon)
      self.adam = adam_optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step)
      # Simple gradient descent op
      gradient_descent = tf.train.GradientDescentOptimizer(self.lr)
      self.sgd = gradient_descent.apply_gradients(
        zip(grads, tvars), global_step=self.global_step)
  
  
  def get_embeddings(self, ids):
    """Performs embedding lookup. Useful as a method for decoder helpers.
       Note this method requires the `embedding_kernel` attribute to be
       declared before being called."""
    return tf.nn.embedding_lookup(self.embedding_kernel, ids)
  
  
  def rnn_cell(self, num_units=None):
    """Get a new RNN cell with wrappers according to the initial config."""
    cell = None
    
    # Allow custom number of hidden units
    if num_units is None:
      num_units = self.hidden_size
    
    # Check to use LSTM or GRU
    if self.use_lstm:
      cell = tf.contrib.rnn.LSTMBlockCell(num_units)
    else:
      cell = tf.contrib.rnn.GRUBlockCell(num_units)
    
    # Check whether to add residual connections
    if self.use_residual:
      cell = tf.contrib.rnn.ResidualWrapper(cell)
    
    # Note: dropout should always be the last wrapper
    if self.dropout < 1:
      cell = tf.contrib.rnn.DropoutWrapper(
        cell, input_keep_prob=self.dropout, output_keep_prob=self.dropout)
    
    return cell
  
  
  def build_encoder(self, encoder_input):
    """Build the RNN stack for the encoder, depending on the initial config."""
    
    # We make only the first encoder layer bidirectional to capture the context
    # (Wu et al., https://arxiv.org/pdf/1609.08144.pdf)
    if self.bidirectional_encoder:
      (encoder_fw_out, encoder_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
        self.rnn_cell(), self.rnn_cell(), encoder_input, dtype=tf.float32,
        sequence_length=self.input_lengths)
      
      # Postprocess the bidirectional output according to the initial config
      if self.bidirectional_mode == 'add':
        encoder_output = encoder_fw_out + encoder_bw_out
      else:
        encoder_output = tf.concat([encoder_fw_out, encoder_bw_out], 2)
        if self.bidirectional_mode == 'project':
          encoder_output = tf.layers.dense(
            encoder_output, self.hidden_size, activation=tf.tanh,
            name='bidirectional_projection')
    else:
      encoder_output, _ = tf.nn.dynamic_rnn(
        self.rnn_cell(), encoder_input, dtype=tf.float32,
        sequence_length=self.input_lengths)
    
    # Only for deep RNN architectures
    if self.rnn_layers > 1:
      encoder_cells = tf.contrib.rnn.MultiRNNCell(
        [self.rnn_cell() for _ in xrange(self.rnn_layers - 1)])
      encoder_output, _ = tf.nn.dynamic_rnn(
        encoder_cells, encoder_output, dtype=tf.float32,
        sequence_length=self.input_lengths)
    
    return encoder_output
  
  
  def build_decoder(self, encoder_output):
    """Build the decoder RNN stack and the final prediction layer."""
    
    decoder_cells = [self.rnn_cell() for _ in range(self.rnn_layers)]
    softmax_layer = Dense(
      self.num_types, name='dense', activation=lambda x: x / self.temperature)
    
    # Decoder for training
    train_decoder = self.get_decoder_instance(
      encoder_output, decoder_cells, softmax_layer)
    train_output = tf.contrib.seq2seq.dynamic_decode(train_decoder)
    
    # Decoder for inference
    infer_decoder = self.get_decoder_instance(
      encoder_output, decoder_cells, softmax_layer, infer=True)
    tf.get_variable_scope().reuse_variables()
    infer_output = tf.contrib.seq2seq.dynamic_decode(
      infer_decoder, maximum_iterations=self.max_decoder_length)
    
    # Returns (training logits, predicted ids for inference)
    return train_output[0].rnn_output, infer_output[0].predicted_ids[:,:,0]
  
  
  def get_decoder_instance(self, encoder_output, cells, dense, infer=False):
    """Return the decoder instance wrapping the built RNNs according to the
       desired mode (training or inference).
       Args:
       `encoder_output`: last layer outputs of the encoder RNN,
       `cells`: list of RNN cell instances,
       `dense`: the output layer,
       Keyword Args:
       `infer`: set to True to return a decoder for inference."""
    
    # These variables need to be tiled for beam search
    batch_size = self.batch_size
    input_lengths = self.input_lengths
    if infer:
      batch_size = self.batch_size * self.beam_size
      input_lengths = tf.contrib.seq2seq.tile_batch(
        input_lengths, multiplier=self.beam_size)
      encoder_output = tf.contrib.seq2seq.tile_batch(
        encoder_output, multiplier=self.beam_size)
    
    # If the model uses attention, wrap one existing layer with the mechanism
    attention_mechanism = None
    if self.attention == 'bahdanau':
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        self.hidden_size, encoder_output, memory_sequence_length=input_lengths)
    elif self.attention == 'luong':
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        self.hidden_size, encoder_output, memory_sequence_length=input_lengths)
    # TODO: maybe allow this to be the last cell
    if self.attention:
      cells[0] = tf.contrib.seq2seq.AttentionWrapper(
        cells[0], attention_mechanism)
    
    # Build the MultiRNNCell if the architecture is deep
    if self.rnn_layers > 1:
      decoder_cell = tf.contrib.rnn.MultiRNNCell(cells)
    
    # Get the initial state of the decoder
    initial_state_pass = encoder_output[:, 0]
    # TODO: maybe allow this to be the last cell
    if self.attention:
      initial_state = tf.contrib.seq2seq.AttentionWrapperState(
        cell_state=initial_state_pass,
        attention=tf.zeros([batch_size, self.hidden_size]),
        alignments=tf.zeros([batch_size, self.max_decoder_length]),
        time=tf.zeros(()), alignment_history=())
    if self.rnn_layers > 1:
      copies = [initial_state_pass for _ in range(self.rnn_layers - 1)]
      initial_state = tuple([initial_state] + copies)
    
    # Training decoder
    if not infer:
      # Prepare training decoder inputs
      decoder_ids = tf.concat(
        [tf.tile([[self.go_id]], [self.batch_size, 1]), self.labels], 1)
      decoder_input = self.get_embeddings(decoder_ids)
      # Build the schedule sampling helper
      train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
        decoder_input, tf.tile([self.max_decoder_length], [batch_size]),
        self.get_embeddings, self.p_sample)
      return tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, train_helper, initial_state, output_layer=dense)
    
    # Inference decoder
    return tf.contrib.seq2seq.BeamSearchDecoder(
      decoder_cell, self.get_embeddings,
      tf.tile([self.go_id], [self.batch_size]), self.eos_id, initial_state,
      self.beam_size, output_layer=dense)
