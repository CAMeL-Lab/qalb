"""This module takes care of all data parsing for the Qatar Arabic Language
   Bank (QALB) dataset released in 2015; including both the L1 dataset of
   corrections from native speakers and the L2 dataset of corrections from
   mistakes made by students of Arabic as a foreign language."""

from __future__ import print_function

import os

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


class WordQALB(BaseDataset):
  """TODO: add descriptive docstring."""
  
  def __init__(self, file_root, sbw='.sbw', **kw):
    """TODO: add descriptive docstring."""
    super(WordQALB, self).__init__(**kw)
    self.file_root = file_root
    # Safe Buckwalter extension
    if not sbw:
      sbw = ''
    self.sbw = sbw
    data_dir = os.path.join('ai', 'datasets', 'data', 'qalb')
    # Prepare training data
    train_input_path = os.path.join(
      data_dir, self.file_root + '.train.sent' + self.sbw
    )
    train_labels = self.maybe_flatten_gold(
      os.path.join(data_dir, self.file_root + '.train')
    )
    with open(train_input_path) as train_file:
      self.train_pairs = self.make_pairs(train_file.readlines(), train_labels)
    # Prepare validation data
    valid_input_path = os.path.join(
      data_dir, self.file_root + '.valid.sent' + self.sbw
    )
    valid_labels = self.maybe_flatten_gold(
      os.path.join(data_dir, self.file_root + '.valid')
    )
    with open(valid_input_path) as valid_file:
      self.valid_pairs = self.make_pairs(valid_file.readlines(), valid_labels)
  
  # TODO: make two different methods for pair making in parent class to avoid
  # child method with different arguments. One method to make the pairs from
  # respective train and validation files, and one to make the pairs from a
  # single file and allowing to specify the ratio of data used for training.
  # pylint: disable=signature-differs
  def make_pairs(self, raw_lines, raw_labels):
    pairs = []
    for raw_line in raw_lines:
      raw_line = raw_line.split()[1:]  # remove document id
      # Convert word n-grams into unique id's
      max_tokens = self.num_steps + self.gram_order - 1
      result = self.tokenize(raw_line[:max_tokens], add_eos=True)
      # Append padding tokens until maximum length is reached
      while len(result) <= self.num_steps:
        result.append(self.type_to_ix['_PAD'])
      # Extract the labels
      pairs.append((result, ))
  
  @classmethod
  def flatten_gold(cls, file_root):
    """Create and return the contents a provided filename that generates a
       parallel corpus to the inputs, following the corrections provided in the
       default gold file m2 format. Note that this step is necessary for
       seq2seq training, and code cannot be borrowed from the evaluation script
       because it never flattens the system output; instead, it finds the
       minimum number of corrections that map the input into the output."""
    with open(file_root + '.m2' + self.sbw) as m2_file:
      raw_m2_data = m2_file.read().split('\n\n')[:-1]  # remove last empty str
    result = []
    for raw_pair in raw_m2_data:
      text = raw_pair.split('\n')[0][2:]  # remove the S marker
      corrections = map(parse_correction, raw_pair.split('\n')[1:])
      result.append(apply_corrections(text, corrections))
    with open(file_root + '.gold' + self.sbw, 'w') as gold_file:
      gold_file.writelines(result)
    return result
