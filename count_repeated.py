"""Analysis specific to repetition errors. This program takes as input the
   output of analysis.py, and will filter errors to only show those related
   to repetition. This is useful since RNNs typically struggle with this, and
   we can show how our preprocessing reduces these types of errors."""

from collections import OrderedDict
import re
import sys


try:
  MIN_REPEATS = sys.argv[1]
  FILENAME = sys.argv[2]

except IndexError:
  print("Usage: python count_repeated.py [min_repeats] [analysis_file]")
  exit()


repeat_re = r'([0-9]+) (PRED|GOLD)\tMISS.+(.+)\3{5,}.*'


misses = OrderedDict()
num_misses = 0
num_pred_misses = 0
with open(FILENAME) as f:
  for line in f:
    match = re.match(repeat_re, line)
    if match:
      num_misses += 1
      if match.group(2) == 'PRED':
        num_pred_misses += 1
      try:
        misses[match.group(1)].append(line)
      except KeyError:
        misses[match.group(1)] = [line]

for lines in misses.values():
  print(*lines, sep='')

print('-' * 80)
print("# pred miss:", num_pred_misses)
print("# gold miss:", num_misses - num_pred_misses)

