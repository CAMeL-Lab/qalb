import io
import sys

import editdistance


with io.open(sys.argv[1], encoding='utf-8') as f:
  proposed_lines = f.readlines()

with io.open(sys.argv[2], encoding='utf-8') as f:
  gold_lines = f.readlines()

try:
  assert len(proposed_lines) == len(gold_lines)
except AssertionError:
  print("Mismatch in number of lines.")
  print("Proposed length:", len(proposed_lines))
  print("Gold length:", len(gold_lines))
  exit()

levs = []
lev_densities = []
for proposed, gold in zip(proposed_lines, gold_lines):
  levs.append(editdistance.eval(proposed, gold))
  lev_densities.append(editdistance.eval(proposed, gold) / len(gold))

print("Levenshtein:", sum(levs) / len(levs))
print("Levenshtein density:", sum(lev_densities) / len(lev_densities))
