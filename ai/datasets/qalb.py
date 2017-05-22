"""This module takes care of all data parsing for the Qatar Arabic Language
   Bank (QALB) dataset released in 2015; including both the L1 dataset of
   corrections from native speakers and the L2 dataset of corrections from
   mistakes made by students of Arabic as a foreign language."""

import os

from ai.datasets import BaseDataset


def parse_correction(correction_line):
  """Parse a raw line describing corrections into data."""
  correction_line = correction_line[2:].split('|||')  # remove the A marker
  start_id, end_id = map(int, correction_line[0].split())
  return (start_id, end_id, correction_line[1], correction_line[2])


class WordQALB(BaseDataset):
  """TODO: add descriptive docstring."""
  
  def __init__(self, **kw):
    super(QALB, self).__init__(**kw)
    
    # Prepare training data
    train_labels = self.maybe_flatten_gold(
      train_gold.readlines('data/qalb/QALB-Train2014')
    )
    with open('data/qalb/QALB-Train2014.sent.sbw') as train_file:
      self.train_pairs = self.make_pairs(train_file.readlines(), train_labels)
    
    # Prepare validation data
    valid_labels = self.maybe_flatten_gold(
      valid_gold.readlines('data/qalb/QALB-Dev2014')
    )
    with open('data/qalb/QALB-Dev2014.sent.sbw') as valid_file:
      self.valid_pairs = self.make_pairs(valid_file.readlines(), valid_labels)
  
  def make_pairs(self, raw_lines, raw_labels):
    pairs = []
    for raw_line in raw_lines:
      raw_line = raw_line.split()[1:]  # remove document id
      # Convert word n-grams into unique id's
      max_tokens = self.num_steps + self.gram_order - 1
      input_ids = self.tokenize(raw_line[:max_tokens], add_eos=True)
      while len(result) <= self.num_steps:
        result.append(self.type_to_ix['_PAD'])
      # Extract the labels
      pairs.append((result, ))
  
  @classmethod
  def maybe_flatten_gold(cls, file_root):
    """Read, or create if it doesn't exist, a provided filename that generates
       a parallel corpus to the inputs, following the corrections provided in
       the default gold file m2 format."""
    gold_filename = file_root + '.gold.sbw'
    if os.path.isfile(gold_filename):
      with open(gold_filename) as gold_file:
        return gold_file.readlines()
    # If the file doesn't exist, it must be created from the m2 file.
    with open(file_root + '.m2.sbw') as m2_file:
      raw_m2_data = m2_file.read().split('\n\n')
    result = []
    for raw_pair in raw_m2_data:
      text = raw_pair.split('\n')[0][2:]  # remove the S marker
      corrections = map(parse_correction, raw_pair.split('\n')[1:])
      # Correct the text
      
      result.append(text)
    with open(gold_filename, 'w') as gold_file:
      gold_file.writelines(result)
    return result
