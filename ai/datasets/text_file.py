"""Reader for generic text files."""

from six.moves import xrange

from ai.datasets import BaseDataset


class TextFile(BaseDataset):
  """Reader for generic text files."""
  
  def __init__(self, filepath, **kw):
    super(TextFile, self).__init__(**kw)
    
    with open(filepath) as text_file:
      raw_data = text_file.read()
    data = []
    
    # Remove the last entry that might have length < `num_steps`.
    max_chars = self.num_steps + self.gram_order - 1
    for i in xrange(0, len(raw_data) - 1, self.num_steps):
      data.append(self.tokenize(raw_data[i:i+max_chars], add_eos=True))
    
    del raw_data  # just for memory efficiency
    self.make_pairs(data)
