"""Character level seq2seq model that supports adding word embeddings."""

import tensorflow as tf
from tensorflow.python.layers.core import Dense

from ai.models import BaseModel


class CharSeq2Seq(BaseModel):
  
  def __init__(self, num_types=0, max_encoder_length=99, max_decoder_length=99,
               pad_id=0, eos_id=1, go_id=2,
               batch_size=32, embedding_size=32, hidden_size=256, rnn_layers=2,
               bidirectional_encoder=False, bidirectional_mode='add',
               use_lstm=False, attention=None, dropout=1., max_grad_norm=5.,
               epsilon=1e-8, beta1=.9, beta2=.999, beam_size=1,
               word_embeddings=None, train_word_embeddings=False, **kw):
    """Build the entire computational graph.
    
    Keyword args:
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
    `bidirectional_encoder`: whether to use a bidirectional encoder RNN,
    `bidirectional_mode`: string for the bidirectional RNN architecture:
      'add' (default): add the forward and backward hidden states,
      'project': use a projection matrix to resize the concatenation of the
                 forward and backward hidden states to `embedding_size`,
      'concat': concatenate the forward and backward inputs and pass that
                as the input to the next RNN,
    `use_lstm`: set to False to use a GRU cell (Cho et al.,
     https://arxiv.org/abs/1406.1078),
    `attention`: 'bahdanau', or 'luong' (none by default),
    `dropout`: keep probability for the non-recurrent connections between RNN
               cells. Defaults to 1.0; i.e. no dropout,
    `max_grad_norm`: clip gradients to maximally this norm,
    `epsilon`: small numerical constant for AdamOptimizer (default 1e-8),
    `beta1`: first order moment decay for AdamOptimizer (default .9),
    `beta2`: second order moment decay for AdamOptimizer (default .999),
    `beam_size`: width of beam search (1=greedy, max=Viterbi),
    `word_embeddings`: a numpy matrix with the word embeddings,
    `train_word_embeddings`: whether to backpropagate on the word embeddings.
    """
    self.num_types = num_types
    self.max_encoder_length = max_encoder_length
    self.max_decoder_length = max_decoder_length
    self.pad_id = pad_id
    self.eos_id = eos_id
    self.go_id = go_id
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.rnn_layers = rnn_layers
    self.bidirectional_encoder = bidirectional_encoder
    self.bidirectional_mode = bidirectional_mode
    self.use_lstm = use_lstm
    self.attention = attention
    self.dropout = dropout
    self.max_grad_norm = max_grad_norm
    self.epsilon = epsilon
    self.beta1 = beta1
    self.beta2 = beta2
    self.beam_size = beam_size
    
    # This is here instead of in `build_graph` to avoid copying unnecessarily
    # the big embedding matrix, as well as to initialize less attributes.
    with tf.variable_scope('embeddings'):
      self.word_embeddings = tf.Variable(
        word_embeddings, trainable=train_word_embeddings, dtype=tf.float32,
        name='word_embeddings')
    
    super().__init__(**kw)
  
  def build_graph(self):
    
    # Allow saving/restoring for learning rate by putting it in the graph.
    self.lr = tf.Variable(
      1e-3, trainable=False, dtype=tf.float32, name='learning_rate')
    
    # Sampling probability variable that can be manually changed. See scheduled
    # sampling paper (Bengio et al., https://arxiv.org/abs/1506.03099)
    self.p_sample = tf.Variable(
      0., trainable=False, dtype=tf.float32, name='sampling_probability')
    
    # Placeholders
    self.inputs = tf.placeholder(
      tf.int32, name='inputs',
      shape=[self.batch_size, self.max_encoder_length, 2])
    self.labels = tf.placeholder(
      tf.int32, name='labels',
      shape=[self.batch_size, self.max_decoder_length])
    # Placeholders for Levenshtein distance summaries
    self.lev = tf.placeholder(tf.float32, name='lev', shape=[])
    self.lev_density = tf.placeholder(tf.float32, name='lev_density', shape=[])
    
    # Sequence lengths - used throughout model
    with tf.name_scope('input_lengths'):
      self.input_lengths = tf.reduce_sum(
        tf.sign(tf.abs(self.inputs[:, :, 0] - self.pad_id)), axis=1)
    
    # Embedding matrix
    with tf.variable_scope('embeddings'):
      sq3 = 3 ** .5  # Uniform(-sqrt3, sqrt3) has variance 1
      self.char_embeddings = tf.get_variable(
        'char_embeddings', [self.num_types, self.embedding_size],
        initializer=tf.random_uniform_initializer(minval=-sq3, maxval=sq3))
    
    with tf.variable_scope('encoder'):
      encoder_output = self.build_encoder(self.get_embeddings(self.inputs))
    
    with tf.variable_scope('decoder'):
      logits, self.generative_output = self.build_decoder(encoder_output)
    
    # Softmax cross entropy loss masked by the target sequence lengths
    with tf.name_scope('loss'):
      mask = tf.cast(tf.sign(self.labels), tf.float32)
      loss = tf.contrib.seq2seq.sequence_loss(logits, self.labels, mask)
    
    # Summaries
    self.perplexity = tf.exp(loss, name='perplexity')
    self.perplexity_summary = tf.summary.scalar('perplexity', self.perplexity)
    self.lev_summary = tf.summary.scalar('lev', self.lev)
    self.lev_density_summary = tf.summary.scalar(
      'lev_density', self.lev_density)
    
    # Index outputs (greedy)
    self.output = tf.argmax(
      logits, axis=2, name='output', output_type=tf.int32)
    
    # Adam optimizer with norm clipping to prevent exploding gradients
    with tf.name_scope('train_ops'):
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, tvars), self.max_grad_norm)
      optimizer = tf.train.AdamOptimizer(
        self.lr, epsilon=self.epsilon, beta1=self.beta1, beta2=self.beta2)
      self.train_step = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step)
  
  def get_embeddings(self, ids):
    """Get the concatenation of word and char embeddings of the given ids."""
    char_embeds = tf.nn.embedding_lookup(self.char_embeddings, ids[:, :, 0])
    word_embeds = tf.nn.embedding_lookup(self.word_embeddings, ids[:, :, 1])
    return tf.concat([char_embeds, word_embeds], -1)
  
  def get_char_embeddings(self, ids):
    """Get only the character embeddings of the given ids.."""
    return tf.nn.embedding_lookup(self.char_embeddings, ids)
 
  def rnn_cell(self, num_units=None, attention_mechanism=None):
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
    
    # Check whether to add an attention mechanism
    if attention_mechanism is not None:
      cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism)
    
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
            encoder_output, self.hidden_size, name='bidirectional_projection')
    else:
      encoder_output, _ = tf.nn.dynamic_rnn(
        self.rnn_cell(), encoder_input, dtype=tf.float32,
        sequence_length=self.input_lengths)
    
    # Only for deep RNN architectures
    if self.rnn_layers > 1:
      cells = [self.rnn_cell() for _ in range(1, self.rnn_layers)]
      encoder_output, _ = tf.nn.dynamic_rnn(
        tf.contrib.rnn.MultiRNNCell(cells), encoder_output, dtype=tf.float32,
        sequence_length=self.input_lengths)
    
    # We only use beam size > 1 during decoding
    if self.beam_size > 1:
      return tf.contrib.seq2seq.tile_batch(encoder_output, self.beam_size)
    return encoder_output
  
  def build_decoder(self, encoder_output):
    """Build the decoder RNN stack and the final prediction layer."""
    beam_batch_size = self.batch_size
    final_encoder_lengths = self.input_lengths
    if self.beam_size > 1:
      beam_batch_size *= self.beam_size
      final_encoder_lengths = tf.contrib.seq2seq.tile_batch(
        final_encoder_lengths, self.beam_size)
    
    # The last RNN layer is wrapped with the attention mechanism
    attention_mechanism = None
    if self.attention == 'bahdanau':
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        self.hidden_size, encoder_output,
        memory_sequence_length=final_encoder_lengths)
    elif self.attention == 'luong':
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        self.hidden_size, encoder_output,
        memory_sequence_length=final_encoder_lengths)
    
    decoder_cell = self.rnn_cell(attention_mechanism=attention_mechanism)
    
    # Use the first output of the encoder to learn an initial decoder state
    initial_state_pass = tf.split(tf.layers.dense(
      encoder_output[:, 0], self.hidden_size * self.rnn_layers,
      activation=tf.tanh, name='initial_decoder_state'
    ), self.rnn_layers, axis=1)
    
    if self.attention:
      initial_state = tf.contrib.seq2seq.AttentionWrapperState(
        cell_state=initial_state_pass[0],
        attention=tf.zeros([beam_batch_size, self.hidden_size]),
        alignments=tf.zeros([beam_batch_size, self.max_decoder_length]),
        time=tf.zeros(()), alignment_history=())
    else:
      initial_state = initial_state_pass[0]
    
    # For deep RNNs, stack the cells and use an initial state that merges the
    # initial state (maybe with attention) with the rest of the learned states
    if self.rnn_layers > 1:
      cells = [self.rnn_cell() for _ in range(1, self.rnn_layers)]
      decoder_cell = tf.contrib.rnn.MultiRNNCell(cells + [decoder_cell])
      initial_state = tuple(list(initial_state_pass[:-1]) + [initial_state])
    
    dense = Dense(self.num_types, name='dense')
    
    # Beam search decoding during inference (greedy for training evals)
    generative_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
      decoder_cell, self.get_char_embeddings,
      tf.tile([self.go_id], [self.batch_size]), self.eos_id, initial_state,
      self.beam_size, output_layer=dense)
    generative_output = tf.contrib.seq2seq.dynamic_decode(
      generative_decoder, maximum_iterations=self.max_decoder_length)
    
    # Training decoder with optional scheduled sampling. If sampling occurs,
    # the output will be fed through the final prediction layer and then fed
    # back to the embedding layer for consistency.
    if self.beam_size == 1:
      tf.get_variable_scope().reuse_variables()
      decoder_input = self.get_char_embeddings(tf.concat(
        [tf.tile([[self.go_id]], [self.batch_size, 1]), self.labels], 1))
      sampling_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
        decoder_input, tf.tile([self.max_decoder_length], [self.batch_size]),
        self.get_char_embeddings, self.p_sample)
      decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, sampling_helper, initial_state, output_layer=dense)
      decoder_output = tf.contrib.seq2seq.dynamic_decode(decoder)
      logits = decoder_output[0].rnn_output
    else:
      # To successfully build the graph just fill the logits with garbage 
      logits = generative_output[0].beam_search_decoder_output.scores
    
    return logits, generative_output[0].predicted_ids[:, :, 0]
