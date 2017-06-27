"""Reader for generic text files."""

from six.moves import xrange

from ai.datasets import BaseDataset
from ai.utils import split_train_test


class TextFile(BaseDataset):
  """Reader for generic text files. Creates character-level LM pairs."""
  
  def __init__(self, filepath, **kw):
    super(TextFile, self).__init__(**kw)
    
    with open(filepath) as text_file:
      raw_data = text_file.read()
    data = []
    
    # Remove the last entry that might have length < `num_steps`.
    max_chars = self.num_steps + self.gram_order
    for i in xrange(0, len(raw_data) - 1, self.num_steps):
      data.append(self.tokenize(raw_data[i:i+max_chars]))
    
    del raw_data  # just for memory efficiency
    self.make_pairs(data)
  
  
  def make_pairs(self, lines):
    pairs = [line[:-1], line[1:] for line in lines]
    self.train_pairs, self.valid_pairs = split_train_test(pairs)
