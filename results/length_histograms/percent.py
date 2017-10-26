"""Utility for seeing how much of the data remains after truncation."""

import re
import sys

try:
  with open(sys.argv[1]) as f:
    lines = list(map(lambda l: re.findall(r'(?<!\d)\d+', l), f.readlines()))
  total = sum(int(l[1]) for l in lines)
  lines = filter(lambda l: int(l[0]) <= int(sys.argv[2]), lines)
  count = sum(int(l[1]) for l in lines)
  print("{0}% ({1} out of {2})".format(100 * count / total, count, total))
except IndexError:
  print("Usage: python percent.py [histogram_filename] [max_length]")
