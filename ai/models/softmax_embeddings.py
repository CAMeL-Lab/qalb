"""This model uses "unsupervised" learning to learn vector representations of
   discrete types. It is meant for language tasks and sequences where the types
   are represented based on their proximity to other tokens. This model uses
   skip-gram and calculates probabilities with a softmax op at every training
   step, and thus is only suitable (and better) for a small number of types."""

import tensorflow as tf

from ai.models import BaseModel


class SoftmaxEmbeddings(BaseModel):
  """TODO: add descriptive docstring."""
  
  def __init__(self, lr=1., num_types=0, embedding_size=256, batch_size=32,
               **kw):
    self.lr = lr
    self.num_types = num_types
    self.embedding_size = embedding_size
    self.batch_size = batch_size
    super(SoftmaxEmbeddings, self).__init__(**kw)
  
  def build_graph(self):
    
    # Placeholders
    self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
    self.labels = tf.placeholder(tf.int32, shape=[self.batch_size])
    
    # Embeddings
    sqrt3 = 3 ** .5  # Uniform(-sqrt3, sqrt3) has variance 1
    embedding_matrix = tf.get_variable(
      'embedding_matrix', [self.num_types, self.embedding_size],
      initializer=tf.random_uniform_initializer(minval=-sqrt3, maxval=sqrt3))
    embeddings = tf.nn.embedding_lookup(embedding_matrix, self.inputs)
    
    # Softmax
    logits = tf.layers.dense(embeddings, self.num_types, name='dense')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.labels, name='sparse_softmax_cross_entropy')
    
    # Train op
    self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)
    
