"""Extract corrected sentences from MADAMIRA output files."""

import re
import sys


try:
  with open(sys.argv[1]) as f:
    sentences = f.read().split('SENTENCE BREAK')
except IndexError:
  print("Usage: python extract_mada.py path/to/example.mada")

with open(sys.argv[1] + '.flat', 'w') as f:
  for sentence in sentences:
    words = re.findall(r';;SVM_PRE[^\n]+\n\*[0-9\.]+\sdiac:([^\s]+)', sentence)
    # Remove diacritics
    new_sentence = ' '.join([re.sub(r'[ًٌٍَُِّْٰ]', '', w) for w in words])
    f.write(new_sentence + '\n')
