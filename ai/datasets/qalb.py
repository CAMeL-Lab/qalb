"""This module takes care of all data parsing for the Qatar Arabic Language
   Bank (QALB) dataset released in 2015; including both the L1 dataset of
   corrections from native speakers and the L2 dataset of corrections from
   mistakes made by students of Arabic as a foreign language."""

import io
import os
import re

import numpy as np

from ai.datasets import BaseDataset


def parse_correction(correction_line):
  """Parse a raw line describing corrections into data, with the format
     (start_id, end_id, correction_type, correction_content). The type can be:
     (1) add_token_before -- insert a token in front of another token
     (2) merge -- merge multiple tokens
     (3) split -- split a token into multiple tokens
     (4) delete_token -- delete a token
     (5) edit -- replace a token with a different token
     (6) move_before -- move a token to a new location in the sentence."""
  correction_line = correction_line[2:].split('|||')  # remove the A marker
  start_id, end_id = map(int, correction_line[0].split())
  return (start_id, end_id, correction_line[1], correction_line[2])


def apply_corrections(text, corrections):
  """Given a text string and a list of corrections in the format returned by
     the `parse_correction` method, return a string of corrected text. Note
     this method is not at all optimized and uses lists."""
  words = text.split()
  # Reverse the corrections to avoid modifying indices (corrections are sorted)
  for start, end, _, content in reversed(corrections):
    # All corrections can be handled with this line (neat!!!)
    words = words[:start] + content.split() + words[end:]
  return ' '.join(words) + '\n'


def max_length_seq(pairs):
  """Get the maximum sequence length of the examples in the provided pairs."""
  return [max(map(len, seq)) for seq in zip(*pairs)]


class QALB(BaseDataset):
  """QALB dataset parsing."""
  
  def __init__(self, file_root, max_input_length=None, max_label_length=None,
               parse_repeated=0, extension='orig', **kw):
    """Arguments:
       `file_root`: the root name of the files in the data/qalb directory.
        The constructor searches for .*.orig, .*.m2, where * is train and dev.
       Keyword arguments:
       `max_input_length`: maximum sequence length for the inputs,
       `max_label_length`: maximum sequence length for the labels,
       `parse_repeated`: e.g. convert `abababab` to `<ab>4`,
       `extension`: name of the non-label file extensions (without the dot),
       Note on usage: to account for the _GO and _EOS tokens that the labels
       have inserted, if the maximum length sequences are in the labels, use
       two extra time steps if the goal is to not truncate anything."""
    super().__init__(**kw)
    self.file_root = file_root
    self.max_input_length = max_input_length
    self.max_label_length = max_label_length
    self.parse_repeated = parse_repeated
    self.extension = extension
    data_dir = os.path.join('ai', 'datasets', 'data', 'qalb')
    # Prepare training data
    train_input_path = os.path.join(
      data_dir, self.file_root + '.train.' + self.extension)
    train_labels = self.maybe_flatten_gold(
      os.path.join(data_dir, self.file_root + '.train'))
    with io.open(train_input_path, encoding='utf-8') as train_file:
      self.train_pairs = self.make_pairs(train_file.readlines(), train_labels)
    # Lock the addition of new characters into the data-- this way, we simulate
    # a real testing environment with possible _UNK tokens.
    self.max_types = self.num_types()
    # Prepare validation data
    valid_input_path = os.path.join(
      data_dir, self.file_root + '.dev.' + self.extension)
    valid_labels = self.maybe_flatten_gold(
      os.path.join(data_dir, self.file_root + '.dev'))
    with io.open(valid_input_path, encoding='utf-8') as valid_file:
      self.valid_pairs = self.make_pairs(valid_file.readlines(), valid_labels)
  
  def untokenize(self, tokens, join_str=''):
    result = super().untokenize(tokens, join_str=join_str)
    if not self.parse_repeated:
      return result
    repl = lambda m: m.group(1)[1:-1] * int(m.group(2))
    return re.sub(r'(<[^>]+>)([0-9]+)', repl, result)
  
  def maybe_flatten_gold(self, file_root, force=False):
    """Create and return the contents a provided filename that generates a
       parallel corpus to the inputs, following the corrections provided in the
       default gold file m2 format. Note that this step is necessary for
       seq2seq training, and code cannot be borrowed from the evaluation script
       because it never flattens the system output; instead, it finds the
       minimum number of corrections that map the input into the output."""
    gold_path = file_root + '.gold'
    if not force and os.path.exists(gold_path):
      with io.open(gold_path, encoding='utf-8') as gold_file:
        return gold_file.readlines()
    print("Flattening labels...")
    m2_path = file_root + '.m2'
    with io.open(m2_path, encoding='utf-8') as m2_file:
      raw_m2_data = m2_file.read().split('\n\n')[:-1]  # remove last empty str
    result = []
    for raw_pair in raw_m2_data:
      text = raw_pair.split('\n')[0][2:]  # remove the S marker
      corrections = map(parse_correction, raw_pair.split('\n')[1:])
      result.append(apply_corrections(text, corrections))
    with io.open(gold_path, 'w', encoding='utf-8') as gold_file:
      gold_file.writelines(result)
    return result
  
  def pad_batch(self, batch):
    """Pad the given batch with zeros."""
    max_input_length = self.max_input_length
    max_label_length = self.max_label_length
    if max_input_length is None or max_label_length is None:
      max_input_length, max_label_length = max_length_seq(batch)
    for i in range(len(batch)):
      while len(batch[i][0]) < max_input_length:
        batch[i][0].append(self.type_to_ix['_PAD'])
      while len(batch[i][1]) < max_label_length:
        batch[i][1].append(self.type_to_ix['_PAD'])
    return batch
  
  # Override this to pad the batches
  def get_train_batches(self, batch_size):
    res = np.array(
      list(map(self.pad_batch, super().get_train_batches(batch_size))))
    #print(np.array(res).shape)
    return res

  def get_valid_batch(self, batch_size):
    """Draw random examples and pad them to the largest sequence drawn."""
    batch = []
    while len(batch) < batch_size:
      sequence = self.valid_pairs[np.random.randint(len(self.valid_pairs))]
      # Optionally discard examples past a maximum input or label length
      input_ok = self.max_input_length is None \
                 or len(sequence[0]) <= self.max_input_length
      label_ok = self.max_label_length is None \
                 or len(sequence[1]) <= self.max_label_length
      if input_ok and label_ok:
        batch.append(sequence)
    return zip(*self.pad_batch(batch))
  
  def shorten_repetitions(self, line):
    """If a pattern is seen at least 2 times contiguously, replace it with
       "pat...pat" (n times) -> "<pat>n"."""
    if not self.parse_repeated or self.parse_repeated < 2:
      return line
    repl = lambda m:'<{}>{}'.format(m.group(1), len(m.group()) // len(m.group(1)))
    return re.sub('(.+?)\1{%d,}' % (self.parse_repeated - 1), repl, line)
  
  def make_pairs(self, input_lines, label_lines):
    pairs = []
    for i in range(len(input_lines)):
      input_line = self.shorten_repetitions(input_lines[i][:-1])  # no newline
      label_line = self.shorten_repetitions(label_lines[i][:-1])  # no newline
      if len(input_line) <= self.max_input_length and \
         len(label_line) <= self.max_label_length - 1:  # eos token
        _input = self.tokenize(input_line)
        label = self.tokenize(label_line)
        label.append(self.type_to_ix['_EOS'])
        pairs.append((_input, label))
    return pairs
