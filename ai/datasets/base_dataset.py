"""TODO: add descriptive module docstring."""

import random

from six.moves import xrange


class BaseDataset(object):
  """Base class for parsing text datasets. Includes defaults for special
     types, tokenizing and untokenizing methods, dataset
     preparation, and batch generation."""
    
  def __init__(self, max_types=None, batch_size=20, num_steps=50, gram_order=1,
               shuffle=False):
    """Keyword arguments:
       `max_types`: an upper bound for the vocabulary size (no bound if falsy),
       `batch_size`: number of lines per training batch,
       `num_steps`: specifies the max length of a line,
       `gram_order`: number of original types in the n-gram,
       `shuffle`: whether to shuffle the list of generated triples."""
    self.max_types = max_types
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.gram_order = gram_order
    self.shuffle = shuffle
    self.train_pairs = []
    self.valid_pairs = []
    # Data structures to tokenize and untokenize data in O(1) time.
    # By default, we add four tokens:
    # '_PAD': padding for examples with variable size and model of fixed size,
    # '_EOS': end of string added before padding to aid the prediction process,
    # '_GO': go token added as the first decoder input for seq2seq models,
    # '_UNK': unknown token used to cover unknown types in the dataset.
    self.ix_to_type = ['_PAD', '_EOS', '_GO', '_UNK']
    # Allow to add any extra defaults without modifying both data structures.
    dictarg = lambda i: [self.ix_to_type[i], i]
    self.type_to_ix = dict(map(dictarg, xrange(len(self.ix_to_type))))
  
  def tokenize(self, input_list):
    """Converts the argument list or string to a list of integer tokens, each
       representing a unique type. If the charachter is not registered, it will
       be added to the `type_to_ix` and `ix_to_type` attributes."""
    result = []
    for i in xrange(len(input_list) - (self.gram_order - 1)):
      gram = tuple(input_list[i:i+self.gram_order])  # lists are unhashable
      if gram not in self.type_to_ix:
        if not self.max_types or self.num_types() < self.max_types:
          self.type_to_ix[gram] = len(self.ix_to_type)
          self.ix_to_type.append(gram)
        else:
          gram = '_UNK'
      result.append(self.type_to_ix[gram])
    return result
  
  def untokenize(self, tokens):
    """Converts the argument list of integer ids back to a string."""
    result = ''
    for t in tokens:
      result += self.ix_to_type[t][0]
    return result
  
  def make_pairs(self, lines, train_data_ratio=.7):
    """Given a list `lines` containing lists of ids, return a list of input
       and target(s) pairs. The percentage of the data used for training can
       be specified with `train_data_ratio`. Override if necessary."""
    make_pairs = lambda line: (line[:-1], line[1:])
    pairs = map(make_pairs, lines)
    num_train_pairs = int(len(lines) * train_data_ratio)
    self.train_pairs = pairs[:num_train_pairs]
    self.valid_pairs = pairs[num_train_pairs:]
  
  def get_batches(self):
    """Groups the triples into batches, and allows randomized order."""
    if self.shuffle:
      random.shuffle(self.train_pairs)
      random.shuffle(self.valid_pairs)
    # Put the triples into batches.
    train_batches = [
      self.train_pairs[i:i+self.batch_size]
      for i in xrange(0, len(self.train_pairs), self.batch_size)
    ]
    valid_batches = [
      self.valid_pairs[i:i+self.batch_size]
      for i in xrange(0, len(self.valid_pairs), self.batch_size)
    ]
    # Prune batches with invalid number of inputs
    train_batches = [b for b in train_batches if len(b) == self.batch_size]
    valid_batches = [b for b in valid_batches if len(b) == self.batch_size]
    return zip(train_batches, valid_batches)
  
  def num_types(self):
    """Return the number of unique n-grams in the dataset."""
    return len(self.ix_to_type)
