"""Beautifier for m2scorer output. This program takes as input a file whose
   content is the output of the m2scorer with the -v flag, and outputs readable
   results that can be easily manipulated with bash commands."""

import codecs
import io
import re
import sys

import numpy as np
import editdistance

from ai.datasets.qalb import apply_corrections


try:
  M2_PATH = sys.argv[1]

except IndexError:
  print("Usage: python analysis.py [m2_output]")
  exit()


def parse_edits(line):
  """Given a line of edits from the m2 file, extract its contents."""
  
  # Get edits inside brackets allowing parentheses
  edit_strings = re.findall(r'\((.*?)\)[,\]]', line)
  
  edits = []
  for edit_string in edit_strings:
    # Splitting by comma is not enough. Some of the edits actually fix spacing
    # when commas are used, so we may can't use edit_string.split(', ')
    # For some reason, the unicode gold strings are enclosed in a list
    m = re.match(r'^(\d+), (\d+), (.*), \[?(.*)\]?$', edit_string)
    edit_items = [m.group(i) for i in range(1, 5)]
    
    # No way to handle this in regex
    if edit_items[3].endswith(']'):
      edit_items[3] = edit_items[3][:-1]
    
    # Cast the indices
    edit_items[0] = int(edit_items[0])
    edit_items[1] = int(edit_items[1])
    
    # Convert unicode-string-inside-string into actual unicode string
    edit_items[2] = codecs.decode(edit_items[2][2:-1], 'unicode_escape') or 'NIL'
    edit_items[3] = codecs.decode(edit_items[3][2:-1], 'unicode_escape') or 'NIL'
    
    edits.append(edit_items)
    
  return edits


def map_inclusion(A, B):
  """Map a -> (a, a in B). Very suboptimal but data is small."""
  
  def _map(a):
    if a in B:
      status = "MATCH"
    else:
      status = "MISS"
    return (a, status)
  
  return map(_map, A)


def f1_score(precision, recall):
  """Computes an F1 score."""
  if precision + recall == 0:
    return 0
  return 2 * precision * recall / (precision + recall)


def beautify_output(m2_output, seq_number):
  """Beautify m2 file output for a single example."""
  
  # Every unit except the first has three extra lines in the beginning
  lines = m2_output.split('\n')[1:]
  if seq_number > 0:
    lines = lines[3:]
  
  # Check for ocassional extra lines
  while lines[0][0] in ['&', '!', '*']:
    lines = lines[1:]
  
  # Remove the first 16 characters of each line-- they are just descriptions
  lines = list(map(lambda l: l[16:], lines))
  
  print("Input:")
  print(lines[0], "\n")
  print("System output:")
  print(lines[1], "\n")
  
  proposed_edits = parse_edits(lines[2])
  gold_edits = parse_edits(lines[3])
  try:
    gold_line = apply_corrections(lines[0], gold_edits)
  except:
    for g in gold_edits:
      print(g)
    raise
  
  print("Gold:")
  print(gold_line)
  
  num_correct_edits = 0
  edit_str = str(seq_number) + ' {}\t{}\t{}-{}\t{}\t{}'
  
  for edit, status in map_inclusion(proposed_edits, gold_edits):
    if status == "MATCH":
      num_correct_edits += 1
    print(edit_str.format("PRED", status, *edit))
  
  print('')
  for edit, status in map_inclusion(gold_edits, proposed_edits):
    print(edit_str.format("GOLD", status, *edit))
  
  eval_str = str(seq_number) + ' {}\t{:10.4f}'
  
  lev = editdistance.eval(lines[1], gold_line)
  lev_density = lev / len(gold_line)
  
  if len(proposed_edits):
    precision = num_correct_edits / len(proposed_edits)
  else:
    precision = int(not len(gold_edits))
  
  if len(gold_edits):
    recall = num_correct_edits / len(gold_edits)
  else:
    recall = int(not len(proposed_edits))
  
  f1 = f1_score(precision, recall)
  
  print('')
  print(eval_str.format('LDIS', lev))
  print(eval_str.format('LDEN', lev_density))
  print('#CORR\t', num_correct_edits)
  print('#PROP\t', len(proposed_edits))
  print('#GOLD\t', len(gold_edits))
  print(eval_str.format('P', precision))
  print(eval_str.format('R', recall))
  print(eval_str.format('F1', f1))
  print("-" * 80)
  
  return np.array([
    lev, lev_density, num_correct_edits, len(proposed_edits), len(gold_edits)])


with io.open(M2_PATH, encoding='utf-8') as f:
  # All the units are separated by 43 dashes. The last element after splitting
  # this way would be the final scores.
  units = f.read().split('-' * 43)
  results = np.zeros(5)
  for i, unit in enumerate(units[:-1]):
    results += beautify_output(unit, i)
  n = len(units) - 1
  eval_str = '{}\t{:10.4f}'
  print(eval_str.format('LDIS', results[0] / n))
  print(eval_str.format('LDEN', results[1] / n))
  
  correct, proposed, gold = list(map(int, results[2:]))
  
  # Avoid divisions by zero
  if proposed:
    p = correct / proposed
  else:
    p = int(not gold)
  if gold:
    r = correct / gold
  else:
    r = int(not proposed)
  
  print('#correct\t', correct)
  print('#proposed\t', proposed)
  print('#gold\t', gold)
  print(eval_str.format('P', p))
  print(eval_str.format('R', r))
  print(eval_str.format('F1', f1_score(p, r)))
