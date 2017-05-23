"""Testing setup for QALB."""

from __future__ import division, print_function

import numpy as np

from ai.datasets import CharQALB, WordQALB


# In an effort to obtain bucket sizes such that the training data is maximally
# balanced within them, we calculate the length of the largest sequences and
# the quartile lengths of the training data with the following script:

# for file_root in ['QALB', 'L2']:
#   for subset in ['train', 'dev']:
#     for extension in ['sent', 'gold']:
#       filename = '{0}.{1}.{2}.sbw'.format(file_root, subset, extension)
#       with open('ai/datasets/data/qalb/' + filename) as data_file:
#         lines = data_file.readlines()
#         start = int(subset == 'sent')
#         word_lengths = map(lambda l: len(l.split()[start:]), lines)
#         char_lengths = map(lambda l: len(' '.join(l.split()[start:])), lines)
        
#         print(
#           "Max sentence word length in {}:".format(filename),
#           max(word_lengths)
#         )
        
#         print("Quartile sentence word lengths in {}:".format(filename))
#         word_lengths = np.sort(word_lengths)
#         for i in [.25, .5, .75, 1.]:
#           quartile = word_lengths[int(len(lines) * i) - 1]
#           print("  {0}%: {1}".format(int(i * 100), quartile))
        
#         print(
#           "Max sentence char length in {}".format(filename),
#           max(char_lengths)
#         )
        
#         print("Quartile sentence char lengths in {}:".format(filename))
#         char_lengths = np.sort(char_lengths)
#         for i in [.25, .5, .75, 1.]:
#           quartile = char_lengths[int(len(lines) * i) - 1]
#           print("  {0}%: {1}".format(int(i * 100), quartile))

# Max sequence lengths for all datasets:
# QALB (word): 185
# QALB (char): 633
# L2 (word): 486 (dev surpasses train; dev.sent=434, dev.gold=484+2)
# L2 (char): 2286 (dev surpasses train; dev.sent=2189, dev.gold=2284+2)

# Quartile sequence lengths for train datasets (inputs and labels):
# QALB (word): (47, 51+2), (53, 56+2), (59, 63+2), (184, 183+2)
# QALB (char): (270, 260+2), (299, 291+2), (331, 324+2), (616, 631+2)
# L2 (word): (82, 87+2), (138, 147+2), (186, 198+2), (361, 377+2)
# L2 (char): (440, 438+2), (741, 742+2), (985, 1010+2), (1918, 2001+2)

# This suggests buckets of the following sizes:

qalb_word_buckets = [(47, 53), (53, 58), (59, 65), (184, 185)]
qalb_char_buckets = [(270, 262), (299, 293), (331, 326), (616, 633)]
l2_word_buckets = [(82, 89), (138, 149), (186, 200), (434, 486)]
l2_char_buckets = [(440, 440), (741, 744), (985, 1012), (2189, 2286)]


print("Building character-level QALB data...")
qalb_char_data = CharQALB('QALB', batch_size=32, buckets=qalb_char_buckets)

#print(map(len, qalb_char_data.train_pairs))
#print(map(len, qalb_char_data.valid_pairs))
batch_inputs, batch_labels = qalb_char_data.get_batches()
for i in range(4):
  print(batch_inputs[i].shape, batch_labels[i].shape)
