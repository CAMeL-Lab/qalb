# pylint: skip-file

"""Reader for WhastApp exported chat history text files."""

import re

from ai.datasets import BaseDataset
from ai.utils import split_train_test


class WhatsAppChats(BaseDataset):
  """Reader for WhastApp exported chat history text files. Creates
     character-level LM pairs."""
  
  def __init__(self, filename='rafi.txt', **kw):
    super().__init__(**kw)
    
    with open('data/whatsapp/' + filename) as chat_history_file:
      raw_lines = chat_history_file.readlines()
    
    max_chars = self.num_steps + self.gram_order - 1
    # pylint: disable=line-too-long
    very_ugly_re = r'[0-9]{1,2}/[0-9]{1,2}/[0-9]{1,2}, [0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2} [AP]M: (?:[^:]+): ([^\r]{1,%s})' % max_chars
    
    # Tokenize and chop lines to `num_steps` limit.
    lines = []
    for raw_line in raw_lines:
      match = re.match(very_ugly_re, raw_line)
      if match:
        line = self.tokenize(match.group(1))
        while len(line) <= self.num_steps:
          line.append(self.type_to_ix['_PAD'])
        lines.append(line)
    
    del raw_lines  # just for memory efficiency
    self.make_pairs(lines)
  
  
  def make_pairs(self, lines):
    pairs = [(line[:-1], line[1:]) for line in lines]
    self.train_pairs, self.valid_pairs = split_train_test(pairs)
