"""Maximum likelihood estimation preprocessor. This program takes as input an m2
   file (should be training data) and any input file, and parses that input file
   with MLE over counts in the m2 file."""

import io
import re
import sys


def count_edit_type(edit_type, lines):
  return sum(map(lambda l: int(bool(re.search(edit_type, l))), lines))


try:
  M2_PATH = sys.argv[1]
  INPUT_PATH = sys.argv[2]

except IndexError:
  print("Usage: python mle.py [train m2] [input]")
  exit()


with io.open(M2_PATH, encoding='utf-8') as f:
  train_examples = f.read().split('\n\n')[:-1]
  counts = {}
  
  # Find all 'Edit' ops and count the occurences per word
  for example in train_examples:
    example = example.split('\n')
    
    # Source sentence is the first, edits are all others
    source = example[0].split()[1:]
    for edit in example[1:]:
      m = re.match(r'A ([0-9]+) [0-9]+\|\|\|Edit\|\|\|([^\|]+)\|', edit)
      if m:
        word = source[int(m.group(1))]
        correction = m.group(2)
        if word not in counts:
          counts[word] = {'_KEEP': 0}
        if correction in counts[word]:
          counts[word][correction] += 1
        else:
          counts[word][correction] = 1
  
  # Count all occurences of word kept
  for example in train_examples:
    source = example.split('\n')[0].split()[1:]
    for word in source:
      if word in counts:
        counts[word]['_KEEP'] += 1
  
  # Take difference to count number of times of word kept as is
  for source, targets in counts.items():
    total = targets['_KEEP']
    not_kept = sum(targets.values()) - total
    counts[source]['_KEEP'] = total - not_kept
    

with io.open(INPUT_PATH, encoding='utf-8') as f:
  for line in f:
    new_line = []
    for word in line.split():
      if word in counts:
        targets = counts[word]
        argmax = max(targets, key=lambda k: targets[k])
        if argmax == '_KEEP':
          argmax = word
        new_line.append(argmax)
      else:
        new_line.append(word)
    print(' '.join(new_line))
