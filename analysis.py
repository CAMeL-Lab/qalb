import codecs
import io
import re
import sys

import editdistance

from ai.datasets.qalb import apply_corrections


def parse_edits(line):
  """Given a line of edits from the m2 file, extract its contents."""
  
  # Get edits inside brackets
  edit_strings = re.findall(r'\(([^\)]*)\)', line)
  
  edits = []
  for edit_string in edit_strings:
    edit_items =  edit_string.split(', ')
    
    # Cast the indices
    edit_items[0] = int(edit_items[0])
    edit_items[1] = int(edit_items[1])
    
    # For some reason, the unicode gold strings are enclosed in a list
    if edit_items[3].startswith('['):
      edit_items[3] = edit_items[3][1:-1]
    
    # Convert unicode-string-inside-string into actual unicode string
    edit_items[2] = codecs.decode(edit_items[2][2:-1], 'unicode_escape')
    edit_items[3] = codecs.decode(edit_items[3][2:-1], 'unicode_escape')
    
    edits.append(edit_items)
    
  return edits


def map_inclusion(A, B):
  """Map a -> (a, a in B)."""
  
  def _map(a):
    if a in B:
      status = "MATCH"
    else:
      status = "MISS"
    return (a, status)
  
  return map(_map, A)


def beautify_output(m2_output, is_first):
  """Beautify m2 file output for a single example."""
  
  # Remove the first 16 characters of each line-- they are just descriptions
  lines = list(map(lambda l: l[16:], m2_output.split('\n')))
  
  # Every unit except the first has three extra lines in the beginning
  if not is_first:
    lines = lines[3:]
  
  print("Input:")
  print(lines[1], "\n")
  print("System output:")
  print(lines[2], "\n")
  
  proposed_edits = parse_edits(lines[3])
  gold_edits = parse_edits(lines[4])
  gold_line = apply_corrections(lines[1], gold_edits)
  
  print("Gold:")
  print(gold_line)
  print("Proposed edits:")
  
  num_correct_edits = 0
  for edit, status in map_inclusion(proposed_edits, gold_edits):
    if status == "MATCH":
      num_correct_edits += 1
    print('PRED\t{0}\t{1}\t{2}\t{3}\t{4}'.format(status, *edit))
  
  print("\nGold edits:")
  for edit, status in map_inclusion(gold_edits, proposed_edits):
    print('GOLD\t{0}\t{1}\t{2}\t{3}\t{4}'.format(status, *edit))
  
  lev = editdistance.eval(lines[2], gold_line)
  lev_density = lev / len(gold_line)
  precision = num_correct_edits / len(proposed_edits)
  recall = num_correct_edits / len(gold_edits)
  f1 = 2 * precision * recall / (precision + recall)
  
  print("\nLevenshtein distance:", lev)
  print(" Levenshtein density:", lev_density)
  print("           Precision:", precision)
  print("              Recall:", recall)
  print("            F1 score:", f1)
  print("-" * 80)
  
  return lev, lev_density, precision, recall, f1


try:
  m2_path = sys.argv[1]
except IndexError:
  print("Usage: python analysis.py [m2_output]")

with io.open(m2_path, encoding='utf-8') as f:
  # All the units are separated by 43 dashes. The last element after splitting
  # this way would be the final scores.
  units = f.read().split('-'  * 43)
  lev, lev_density, precision, recall, f1 = [0] * 5
  for unit in units[:-1]:
    _lev, _lev_density, _precision, _recall, _f1 = beautify_output(
      unit, unit == units[0])
    lev += _lev
    lev_density += _lev_density
    precision += _precision
    recall += _recall
    f1 += _f1
  
  n = len(units) - 1
  print("Levenshtein distance:", lev / n)
  print(" Levenshtein density:", lev_density / n)
  print("           Precision:", precision / n)
  print("              Recall:", recall / n)
  print("            F1 score:", f1 / n)
