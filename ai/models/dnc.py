# pylint: skip-file

"""This module is still in progress, and will contain multiple implementations
   for the differentiable neural computer (Graves et al., 2016)."""

from six.moves import xrange
import tensorflow as tf

from ai.models import BaseModel


def oneplus(x):
  return 1 + tf.log(1 + tf.exp(-x))

def softcos(M, k, beta):
  cos_distance = tf.reduce_sum(M * k, axis=-1) / \
    (tf.norm(M, axis=-1) * tf.norm(k, axis=-1))
  return tf.nn.softmax(beta * cos_distance)


class DNC(Model):
  """TODO: add descriptive docstring."""
  
  def __init__(self, num_rnn_units=50, num_rnn_layers=1, batch_size=10,
    time_steps=20, num_memory_units=40, memory_unit_size=60, num_read_heads=2,
    output_dim=10, **kw):
    """TODO: add documentation for the keyword arguments."""
    self.num_rnn_units = num_rnn_units
    self.num_rnn_layers = num_rnn_layers
    self.time_steps = time_steps
    self.num_memory_units = num_memory_units
    self.memory_unit_size = memory_unit_size
    self.num_read_heads = num_read_heads
    self.output_dim = output_dim
    # Initialize inputs and labels with placeholders in constructor to allow
    # extensions to the graph both before and after the DNC architecture
    self._inputs = tf.placeholder(tf.float32, name='default_inputs',
      shape=[self.time_steps, self.num_rnn_units])
    self._labels = tf.placeholder(tf.float32, name='default_labels',
      shape=[self.time_steps, self.output_dim])
    super(DNC, self).__init__(**kw)
  
  def build_graph(self):
    X = self.num_rnn_units
    N = self.num_memory_units
    W = self.memory_unit_size
    R = self.num_read_heads
    Y = self.output_dim
    
    with tf.variable_scope('DNC'):
      
      with tf.variable_scope('controller'):
        rnn_cell = tf.contrib.rnn.MultiRNNCell(tf.contrib.rnn.LSTMBlockCell(X)
          for _ in xrange(self.num_rnn_layers))
        state = tf.zeros(rnn_cell.state_size[0])
        
      """
      with tf.variable_scope('interface'):
        IRK = tf.get_variable('read_key_matrix', [X, R * W],
          initializer=tf.zeros_initializer())
        IRS = tf.get_variable('read_stength_matrix', [X, R],
          initializer=tf.zeros_initializer())
        IWK = tf.get_variable('write_key_matrix', [X, W],
          initializer=tf.zeros_initializer())
        IWS = tf.get_variable('write_strength_vector', [X, 1],
          initializer=tf.zeros_initializer())
        IEM = tf.get_variable('erase_matrix', [X, W],
          initializer=tf.zeros_initializer())
        IWM = tf.get_variable('write_matrix', [X, W],
          initializer=tf.zeros_initializer())
        IFG = tf.get_variable('free_gate_matrix', [X, R],
          initializer=tf.zeros_initializer())
        IAG = tf.get_variable('allocation_gate_vector', [X, 1],
          initializer=tf.zeros_initializer())
        IWG = tf.get_variable('write_gate_vector', [X, 1],
          initializer=tf.zeros_initializer())
        IRM = tf.get_variable('read_mode_matrix', [X, 3],
          initializer=tf.zeros_initializer())
      """
      
      with tf.name_scope('memory'):
        M = tf.get_variable('memory', [N, W], trainable=False,
          initializer=tf.zeros_initializer())
      
      with tf.name_scope('memory_allocation'):
        memory_usage = tf.zeros([N])
        write_weighting = tf.zeros([N])
      
      with tf.name_scope('memory_linkage'):
        # TODO: use a sparse implementation for L
        L = tf.zeros([N, N])  # memory linkage matrix
        precedence_weightings = tf.zeros(N)
        read_weightings = tf.zeros([R, N])
      
      with tf.variable_scope('read'):
        VM = tf.get_variable('output_matrix', [X, Y],
          initializer=tf.zeros_initializer())
        RM = tf.get_variable('read_matrix', [R * W, Y],
          initializer=tf.zeros_initializer())
      
      outputs = []
      for t in xrange(self.time_steps):
        with tf.variable_scope('time_step') as time_step_scope:
          
          with tf.variable_scope('controller'):
            # TODO: allow larger batch size
            rnn_out, state = rnn_cell(tf.reshape(self._inputs[t], [1, X]), state)
          
          with tf.name_scope('interface'):
            
            # Get all the interface vectors emmited by the controller
            if t > 1:
              time_step_scope.reuse_variables()
            im_shape = [X, R * W + 3 * W + 5 * R + 3]
            interface_matrix = tf.get_variable('interface_matrix', im_shape,
              initializer=tf.zeros_initializer())
            # TODO: finish this with tf.dynamic_partition
            tf.matmul(rnn_out, interface_matrix)
            
            # TODO: allow larger batch size
            """
            read_keys = tf.reshape(tf.matmul(rnn_out, IRK), [R, W])
            read_strengths = tf.reshape(oneplus(tf.matmul(rnn_out, IRS)), [R])
            write_key = tf.matmul(rnn_out, IWK)
            write_strength = oneplus(tf.matmul(rnn_out, IWS))
            erase_vector = tf.nn.sigmoid(tf.matmul(rnn_out, IEM))
            write_vector = tf.matmul(rnn_out, IWM)
            free_gates = tf.reshape(tf.nn.sigmoid(tf.matmul(rnn_out, IFG)), [R])
            allocation_gate = tf.nn.sigmoid(tf.matmul(rnn_out, IAG))
            write_gate = tf.nn.sigmoid(tf.matmul(rnn_out, IWG))
            read_modes = tf.reshape(tf.nn.softmax(tf.matmul(rnn_out, IRM)), [3])
            """
          
          with tf.name_scope('memory_allocation'):
            memory_retention = tf.ones([N])
            # TODO: more tensorflowy way of doing this?
            for i in xrange(R):
              memory_retention *= 1 - free_gates[i] * read_weightings[i]
            memory_usage = memory_retention * (memory_usage + \
              write_weighting - memory_usage * write_weighting)
            # Indices of sorted memory usage in ascending order
            s = tf.nn.top_k(-memory_usage, k=N).indices
            # TODO: more tensorflowy way of doing this?
            collected_usage = 1.
            allocation_weighting = tf.zeros([N])
            for i in xrange(N):
              allocation_weighting += tf.scatter_nd([[s[i]]],
                [collected_usage * (1 - memory_usage[s[i]])], [N])
              collected_usage *= memory_usage[s[i]]
            # The content-based addressing is based on similarity with
            # existing entries in the memory
            write_content_weighting = softcos(M, write_key, write_strength)
            write_weighting = (allocation_gate * allocation_weighting + \
              (1 - allocation_gate) * write_content_weighting) * write_gate
            write_weighting = tf.reshape(write_weighting, [N])
          
          with tf.name_scope('write'):
            w = tf.reshape(write_weighting, [N, 1])
            M *= 1 - tf.matmul(w, tf.reshape(erase_vector, [1, W]))
            M += tf.matmul(w, tf.reshape(write_vector, [1, W]))
          
          with tf.name_scope('memory_linkage'):
            precedence_weightings *= (1 - tf.reduce_sum(write_weighting))
            precedence_weightings += write_weighting
            # TODO: use a sparse implementation for L
            w_tile = tf.tile(w, [1, N])
            L *= (1 - w_tile - tf.transpose(w_tile))
            L += tf.matmul(w, tf.reshape(precedence_weightings, [1, N]))
          
          with tf.name_scope('read'):
            forward_weightings = tf.matmul(read_weightings, L)
            backward_weightings = tf.matmul(read_weightings, tf.transpose(L))
            read_content_weightings = tf.stack([softcos(M, read_keys[i],
              read_strengths[i]) for i in xrange(R)])
            read_weightings = read_modes[0] * backward_weightings + \
                              read_modes[1] * read_content_weightings + \
                              read_modes[2] * forward_weightings
            reads = tf.reshape(tf.matmul(read_weightings, M), [1, R * W])
          
          outputs.append(tf.matmul(rnn_out, VM) + tf.matmul(reads, RM))
      
      self._outputs = tf.reshape(tf.stack(outputs), [self.time_steps, Y])
