"""Maximum likelihood estimation preprocessor.

This program takes as input an m2 file (should be training data) and any input
file, and parses that input file with MLE over counts in the m2 file. For
cleanup purposes it removes kashidas as a preprocessing step (empirically found
to be better than removing them after the MLE).
"""

import io
import re
import sys


KASHIDA = u'\u0640'
EDIT_RE = r'A ([0-9]+) ([0-9]+)\|\|\|[^\|]+\|\|\|([^\|]*)'


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
      m = re.match(EDIT_RE, edit)
      if m:
        try:
          word = source[int(m.group(1))]
        except IndexError:
          pass
        # This tuple has format (id_difference, correction_str)
        correction = (int(m.group(2)) - int(m.group(1)), m.group(3))
        
        if word not in counts:
          counts[word] = {'_KEEP': 0}
        if correction in counts[word]:
          counts[word][correction] += 1
        else:
          counts[word][correction] = 1
  
  # Count all occurences of word
  for example in train_examples:
    source = example.split('\n')[0].split()[1:]
    for word in source:
      if word in counts:
        counts[word]['_KEEP'] += 1
  
  # Take difference to count number of times of word kept as is
  for word, corrections in counts.items():
    total = corrections['_KEEP']
    not_kept = sum(corrections.values()) - total
    counts[word]['_KEEP'] = total - not_kept
    

with io.open(INPUT_PATH, encoding='utf-8') as f:
  for line in f:
    line = line.replace(KASHIDA, '')
    new_line = line.split()
    # We want an analog to "reverse(enumerate(line.split()))"
    i = len(new_line)
    for word in reversed(line.split()):
      i -= 1
      if word in counts:
        targets = counts[word]
        correction = max(targets, key=lambda k: targets[k])  # argmax
        if correction != '_KEEP':
          diff = correction[0]
          correction_list = correction[1].split()
          new_line = new_line[:i] + correction_list + new_line[i + diff:]
    print(' '.join(new_line))
