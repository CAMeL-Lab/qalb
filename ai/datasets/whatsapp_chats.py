import re

from base_datasets import Dataset


class WhatsAppChats(Dataset):
  """Reader for WhastApp exported chat history text files."""
  
  def __init__(self, filename='rafi.txt', **kw):
    super(WhatsAppChats, self).__init__(**kw)
    with open('data/whatsapp/' + filename) as f:
      raw_lines = f.readlines()
    
    max_chars = self.num_steps + self.gram_order - 1
    very_ugly_re = r'[0-9]{1,2}/[0-9]{1,2}/[0-9]{1,2}, [0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2} [AP]M: (?:[^:]+): ([^\r]{1,%s})' % max_chars
    
    # Tokenize and chop lines to `num_steps` limit.
    lines = []
    for raw_line in raw_lines:
      match = re.match(very_ugly_re, raw_line)
      if match:
        line = self.tokenize(match.group(1), add_eos=True)
        while len(line) <= self.num_steps:
          line.append(self.char_to_ix['_PAD'])
        lines.append(line)
    
    del raw_lines  # just for memory efficiency
    self.make_triples(lines)
