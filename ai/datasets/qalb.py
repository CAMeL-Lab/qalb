"""This module takes care of all data parsing for the Qatar Arabic Language
   Bank (QALB) dataset released in 2015; including both the L1 dataset of
   corrections from native speakers and the L2 dataset of corrections from
   mistakes made by students of Arabic as a foreign language."""

import io
import os

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
               extension='', **kw):
    """Arguments:
       `file_root`: the root name of the files in the data/qalb directory.
        The constructor searches for .*.orig, .*.m2, where * is train and dev.
       Keyword arguments:
       `max_input_length`: maximum sequence length for the inputs,
       `max_label_length`: maximum sequence length for the labels,
       `extension`: name of the data file extensions,
       Note on usage: to account for the _GO and _EOS tokens that the labels
       have inserted, if the maximum length sequences are in the labels, use
       two extra time steps if the goal is to not truncate anything."""
    super().__init__(**kw)
    self.file_root = file_root
    self.max_input_length = max_input_length
    self.max_label_length = max_label_length
    self.extension = extension
    data_dir = os.path.join('ai', 'datasets', 'data', 'qalb')
    # Prepare training data
    train_input_path = os.path.join(
      data_dir, self.file_root + '.train.orig' + self.extension
    )
    train_labels = self.maybe_flatten_gold(
      os.path.join(data_dir, self.file_root + '.train')  # method completes it
    )
    with io.open(train_input_path, encoding='utf-8') as train_file:
      self.train_pairs = self.make_pairs(train_file.readlines(), train_labels)
    self.max_train_lengths = max_length_seq(self.train_pairs)
    # Prepare validation data
    valid_input_path = os.path.join(
      data_dir, self.file_root + '.dev.orig' + self.extension
    )
    valid_labels = self.maybe_flatten_gold(
      os.path.join(data_dir, self.file_root + '.dev')
    )
    with io.open(valid_input_path, encoding='utf-8') as valid_file:
      self.valid_pairs = self.make_pairs(valid_file.readlines(), valid_labels)
    self.max_valid_lengths = max_length_seq(self.valid_pairs)
  
  # Override to set the default joining string to not be a whitespace.
  def untokenize(self, tokens, join_str=''):
    return super().untokenize(tokens, join_str=join_str)
  
  def maybe_flatten_gold(self, file_root, force=False):
    """Create and return the contents a provided filename that generates a
       parallel corpus to the inputs, following the corrections provided in the
       default gold file m2 format. Note that this step is necessary for
       seq2seq training, and code cannot be borrowed from the evaluation script
       because it never flattens the system output; instead, it finds the
       minimum number of corrections that map the input into the output."""
    gold_path = file_root + '.gold' + self.extension
    if not force and os.path.exists(gold_path):
      with io.open(gold_path, encoding='utf-8') as gold_file:
        return gold_file.readlines()
    print("Flattening labels...")
    m2_path = file_root + '.m2' + self.extension
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
  
  def make_pair(self, input_line, label_line):
    """Given an input and label in text or list form, convert the n-grams to
       their unique type id's. This also takes care of padding and adding
       other tokens that are helpful for the decoder RNN. If the arguments are
       strings, the `tokenize` method will iterate over the strings resulting
       in character-level types. If they are iterables instead, the method will
       use the elements (or their n-grams) as their types."""
    input_ids = self.tokenize(input_line)
    label_ids = self.tokenize(label_line)
    label_ids.append(self.type_to_ix['_EOS'])
    return input_ids, label_ids
  
  def get_batch(self, batch_size, draw_from_valid=False):
    """Draw random examples and pad them to the largest sequence drawn.
       The batch can be drawn from the validation set if the keyowrd argument
       `draw_from_valid` is set to True."""
    batch = []
    while len(batch) < batch_size:
      if draw_from_valid:
        sequence = self.valid_pairs[np.random.randint(len(self.valid_pairs))]
      else:
        sequence = self.train_pairs[np.random.randint(len(self.train_pairs))]
      # Optionally discard examples past a maximum input or label length
      input_ok = self.max_input_length is None \
                 or len(sequence[0]) <= self.max_input_length
      label_ok = self.max_label_length is None \
                 or len(sequence[1]) <= self.max_label_length
      if input_ok and label_ok:
        batch.append(sequence)
    for i in range(batch_size):
      max_input_length = self.max_input_length
      max_label_length = self.max_label_length
      if max_input_length is None or max_label_length is None:
        max_input_length, max_label_length = max_length_seq(batch)
      while len(batch[i][0]) < max_input_length:
        batch[i][0].append(self.type_to_ix['_PAD'])
      while len(batch[i][1]) < max_label_length:
        batch[i][1].append(self.type_to_ix['_PAD'])
    return zip(*batch)  # return as (batch_of_inputs, batch_of_labels)
  
  def make_pairs(self, input_lines, label_lines):
    pairs = []
    for i in range(len(input_lines)):
      input_line = input_lines[i]
      label_line = label_lines[i][:-1]  # remove newline
      pairs.append(self.make_pair(input_line, label_line))
    return pairs
