"""Testing setup for QALB."""

from __future__ import division, print_function

from collections import deque

import numpy as np
import tensorflow as tf

from ai.datasets import qalb
from ai.models import Seq2Seq


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

L1_WORD_BUCKETS = [(47, 53), (53, 58), (59, 65), (184, 185)]
L1_CHAR_BUCKETS = [(270, 262), (299, 293), (331, 326), (616, 633)]
L2_WORD_BUCKETS = [(82, 89), (138, 149), (186, 200), (434, 486)]
L2_CHAR_BUCKETS = [(440, 440), (741, 744), (985, 1012), (2189, 2286)]


class DynamicWordQALB(qalb.DynamicQALB, qalb.WordQALB):
  pass


### HYPERPARAMETERS
LR = 1e-3
LR_DECAY = .9
NUM_BAD_TO_DECAY = 5
BATCH_SIZE = 64
EMBEDDING_SIZE = 256
RNN_LAYERS = 1
BIDIRECTIONAL_ENCODER = True
MAX_GRAD_NORM = 5.
RNN_CELL = tf.contrib.rnn.LSTMBlockCell
USE_LUONG_ATTENTION = True
SAMPLING_PROBABILITY = 0.


### CONFIG
MAX_TYPES = 20000  # set to `None` to keep all the unique types
MAX_ENCODER_LENGTH = 50
MAX_DECODER_LENGTH = 50
NUM_STEPS_PER_EVAL = 10
NUM_STEPS_PER_SAVE = 100
NUM_BAD_TO_STOP = 50
RESTORE = False
MODEL_NAME = 'qalb-word-sampling-0'


print("Building dynamic word-level QALB data...")
dataset = DynamicWordQALB('QALB', max_types=MAX_TYPES, batch_size=BATCH_SIZE,
                          max_input_length=MAX_ENCODER_LENGTH,
                          max_label_length=MAX_DECODER_LENGTH)

print("Building computational graph...")
graph = tf.Graph()
with graph.as_default():
  m = Seq2Seq(num_types=dataset.num_types(),
              max_encoder_length=MAX_ENCODER_LENGTH,
              max_decoder_length=MAX_DECODER_LENGTH,
              pad_id=dataset.type_to_ix['_PAD'],
              go_id=dataset.type_to_ix['_GO'], lr=LR, lr_decay=LR_DECAY,
              batch_size=BATCH_SIZE, embedding_size=EMBEDDING_SIZE,
              rnn_layers=RNN_LAYERS,
              bidirectional_encoder=BIDIRECTIONAL_ENCODER,
              max_grad_norm=MAX_GRAD_NORM, rnn_cell=RNN_CELL,
              use_luong_attention=USE_LUONG_ATTENTION,
              sampling_probability=SAMPLING_PROBABILITY, restore=RESTORE,
              model_name=MODEL_NAME)

with tf.Session(graph=graph) as sess:
  m.start()
  
  # Recent improvements tracked for learning rate decay
  num_passed_lr = 0
  average_lr = 0
  last_perplexities_lr = deque(NUM_BAD_TO_DECAY * [0.], NUM_BAD_TO_DECAY)
  
  # Recent improvements tracked for early termination
  num_passed_stop = 0
  average_stop = 0
  last_perplexities_stop = deque(NUM_BAD_TO_STOP * [0.], NUM_BAD_TO_STOP)
  
  print("Entering training loop...")
  while True:
    train_inputs, train_labels = dataset.get_batch()
    train_fd = {m.inputs: train_inputs, m.labels: train_labels}
    sess.run(m.train_op, feed_dict=train_fd)
    step = m.global_step.eval()
    
    if step % NUM_STEPS_PER_EVAL == 0:
      # Show learning rate and sample outputs from training set
      lr, train_ppx, train_output, train_summary = sess.run(
        [m._lr, m.perplexity, m.output, m.summary_op], feed_dict=train_fd
      )
      # Evaluate and show samples on validation set
      valid_inputs, valid_labels = dataset.get_batch(draw_from_valid=True)
      valid_fd = {m.inputs: valid_inputs, m.labels: valid_labels}
      valid_ppx, valid_output, valid_summary = sess.run(
        [m.perplexity, m.output, m.summary_op], feed_dict=valid_fd
      )
      m.train_writer.add_summary(train_summary, global_step=step)
      m.valid_writer.add_summary(valid_summary, global_step=step)
      print("================================================================")
      print("Step {0} (lr={1}, train_ppx={2}, valid_ppx={3})".format(
        step, str(lr)[:6], train_ppx, valid_ppx
      ))
      print("================================================================")
      print("Sample train input:")
      print(dataset.untokenize(train_inputs[0]))
      print("Sample train target:")
      print(dataset.untokenize(train_labels[0]))
      print("Sample train output:")
      print(dataset.untokenize(train_output[0]))
      print("Sample valid input:")
      print(dataset.untokenize(valid_inputs[0]))
      print("Sample valid target:")
      print(dataset.untokenize(valid_labels[0]))
      print("Sample valid output:")
      print(dataset.untokenize(valid_output[0]))
      
      # Decay the learning rate if there has been no recent improvement
      last_perplexities_lr.appendleft(train_ppx)
      new_average = sum(last_perplexities_lr) / NUM_BAD_TO_DECAY
      if num_passed_lr == NUM_BAD_TO_DECAY:
        if new_average > average_lr:
          num_passed_lr = 0
          sess.run(m.decay_lr)
      else:
        num_passed_lr += 1
      average_lr = new_average
      
      # Stop if there has been no improvement for a while
      last_perplexities_stop.appendleft(valid_ppx)
      new_average = sum(last_perplexities_stop) / NUM_BAD_TO_STOP
      if num_passed_stop == NUM_BAD_TO_STOP:
        if new_average > average_stop:
          num_passed_stop = 0
          exit()
      else:
        num_passed_stop += 1
      average_stop = new_average
    
    if step % NUM_STEPS_PER_SAVE == 0:
      m.save()
