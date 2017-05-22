"""This module takes care of all data parsing for the Qatar Arabic Language
   Bank (QALB) dataset released in 2015; including both the L1 dataset of
   corrections from native speakers and the L2 dataset of corrections from
   mistakes made by students of Arabic as a foreign language."""

from __future__ import print_function

import os

from six.moves import xrange

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


class BaseQALB(BaseDataset):
  """Base class for QALB data parsing. This parent class takes care of reading
     and preprocessing the data files, and children classes can specify
     different tokenizations (word-based, character-based, etc)."""
  
  def __init__(self, file_root, sbw='.sbw', **kw):
    """Arguments:
       `file_root`: the root name of the files in the data/qalb directory.
        The constructor searches for .*.sent, .*.m2, where * is train and dev.
       Keyword arguments:
       `sbw`: name of the safe Buckwalter extension (or none if it's falsy).
       Note on usage: to account for the _GO and _EOS tokens that the labels
       have inserted, if the maximum length sequences are in the labels, use
       two extra time steps if the goal is to not truncate anything."""
    super(BaseQALB, self).__init__(**kw)
    self.file_root = file_root
    if not sbw:
      sbw = ''
    self.sbw = sbw
    data_dir = os.path.join('ai', 'datasets', 'data', 'qalb')
    # Prepare training data
    train_input_path = os.path.join(
      data_dir, self.file_root + '.train.sent' + self.sbw
    )
    train_labels = self.flatten_gold(
      os.path.join(data_dir, self.file_root + '.train')  # method completes it
    )
    with open(train_input_path) as train_file:
      self.train_pairs = self.make_pairs(train_file.readlines(), train_labels)
    # Prepare validation data
    valid_input_path = os.path.join(
      data_dir, self.file_root + '.dev.sent' + self.sbw
    )
    valid_labels = self.flatten_gold(
      os.path.join(data_dir, self.file_root + '.dev')
    )
    with open(valid_input_path) as valid_file:
      self.valid_pairs = self.make_pairs(valid_file.readlines(), valid_labels)
  
  # TODO: make two different methods for pair making in parent class to avoid
  # child method with different arguments. One method to make the pairs from
  # respective train and validation files, and one to make the pairs from a
  # single file and allowing to specify the ratio of data used for training.
  # pylint: disable=signature-differs
  def make_pairs(self, text_lines, text_labels):
    pass
  
  def make_pair(self, input_line, label_line):
    """Given an input and label in text or list form, convert the n-grams to
       their unique type id's. This also takes care of padding and adding
       other tokens that are helpful for the decoder RNN. If the arguments are
       strings, the `tokenize` method will iterate over the strings resulting
       in character-level types. If they are iterables instead, the method will
       use the elements (or their n-grams) as their types."""
    # This already takes care of making the n-grams their own unique types.
    input_ids = self.tokenize(input_line)
    # Optimization with append instead of concatenate. Necessary when
    # working at the character level as the arrays get very long.
    label_ids = [self.type_to_ix['_GO']]
    _label_ids = self.tokenize(label_line)
    for label_id in _label_ids:
      label_ids.append(label_id)
    label_ids.append(self.type_to_ix['_EOS'])
    # Append padding tokens until maximum length is reached
    while len(input_ids) < self.num_steps:
      input_ids.append(self.type_to_ix['_PAD'])
    while len(label_ids) < self.num_steps:
      label_ids.append(self.type_to_ix['_PAD'])
    return input_ids, label_ids
  
  def flatten_gold(self, file_root):
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


class CharQALB(BaseQALB):
  """Character-level data parser for QALB dataset."""
  
  def make_pairs(self, input_lines, label_lines):
    pairs = []
    for i in xrange(len(input_lines)):
      # Remove the document id (first word) and truncate to max char length
      input_line = ' '.join(input_lines[i].split()[1:])[:self.num_steps]
      label_line = label_lines[i][:self.num_steps-2]
      pairs.append(self.make_pair(input_line, label_line))
    return pairs


class WordQALB(BaseQALB):
  """Word-level data parser for QALB dataset."""
  
  def make_pairs(self, input_lines, label_lines):
    pairs = []
    for i in xrange(len(input_lines)):
      # Compensate for document id
      input_line = input_lines[i].split()[1:self.num_steps+1]
      label_line = label_lines[i].split()[:self.num_steps-2]
      pairs.append(self.make_pair(input_line, label_line))
    return pairs
