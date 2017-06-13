"""Testing setup for NextGram model over text datasets."""

from __future__ import division, print_function

from collections import deque

import numpy as np
import tensorflow as tf

from ai.datasets import TextFile
from ai.models import NextGram


# HYPERPARAMETERS
LR = 5e-4
LR_DECAY = .9
NUM_BAD = 10
BATCH_SIZE = 32
NUM_STEPS = 32
EMBED_SIZE = 200
NUM_RNN_LAYERS = 2
MAX_GRAD_NORM = 5.
RNN_CELL = tf.contrib.rnn.GRUBlockCell
GRAM_ORDER = 1
SHUFFLE_DATA = True

# Other config
STEPS_PER_EVAL = 50
STEPS_PER_SAVE = 500
SHOULD_RESTORE = True
MODEL_NAME = 'bigram-bible-3l'


def train(m, sess):
  # Recent improvements tracked for learning rate decay
  # pylint: disable=invalid-name
  num_passed = 0
  average = 0
  last_perplexities = deque(NUM_BAD * [0.], NUM_BAD)
  # Training loop
  print("Entering training loop (step {})".format(m.global_step.eval()))
  while True:
    for train_batch, valid_batch in DATASET.get_batches():
      # Construct feed dicts
      train_inputs, train_labels = zip(*train_batch)
      train_fd = {m.inputs: train_inputs, m.labels: train_labels}
      valid_inputs, valid_labels = zip(*valid_batch)
      valid_fd = {m.inputs: valid_inputs, m.labels: valid_labels}
      # OPTIONAL: stop after the specified number of steps
      step = m.global_step.eval()
      if step > 5000:
        exit()
      # Training step
      sess.run(m.train_op, feed_dict=train_fd)
      # Evaluation every specified number of steps
      if step % STEPS_PER_EVAL == 0:
        # Save train stats
        lr, train_perp, train_summary = sess.run(
          [m.lr, m.perplexity, m.summary_op], feed_dict=train_fd
        )
        m.train_writer.add_summary(train_summary, global_step=step)
        # Save valid stats, and generate some sample output
        output, valid_perp, valid_summary = sess.run(
          [m.output, m.perplexity, m.summary_op], feed_dict=valid_fd
        )
        m.valid_writer.add_summary(valid_summary, global_step=step)
        # Print generated samples
        print("=========================================================")
        print("Step {0} (lr={1}, train_perp={2}, valid_perp={3})".format(
          step, str(lr)[:6], str(train_perp)[:6], str(valid_perp)[:6]
        ))
        print("=========================================================")
        print("Sample output:")
        print(DATASET.untokenize([valid_inputs[0][0]], join_str='') + \
              DATASET.untokenize(output, join_str=''))
        # Decay the learning rate if there has been no recent improvement
        last_perplexities.appendleft(train_perp)
        new_average = sum(last_perplexities) / NUM_BAD
        if num_passed == NUM_BAD:
          if new_average > average:
            num_passed = 0
            sess.run(m.decay_lr)
        else:
          num_passed += 1
        average = new_average
      # Save the trainable variables every specified number of steps
      if step % STEPS_PER_SAVE == 0:
        m.save()


def hallucinate(m, sess):
  for t in np.arange(.1, 10., .1):
    seed = DATASET.tokenize('LO')[0]
    decoder_out = sess.run(m.generative_output, feed_dict={m.seed: seed, m.temperature: t})[0]
    print(t)
    print('L' + DATASET.untokenize(decoder_out.predicted_ids[0][:, 0], join_str=''))
    print('\n')

if __name__ == '__main__':
  # Read and preprocess data
  print("Reading data...")
  DATASET = TextFile('ai/datasets/data/bible.txt', batch_size=BATCH_SIZE,
                     num_steps=NUM_STEPS, gram_order=GRAM_ORDER,
                     shuffle=SHUFFLE_DATA)
  # Build the computational graph
  GRAPH = tf.Graph()
  with GRAPH.as_default():
    # pylint: disable=too-many-locals,invalid-name
    m = NextGram(num_types=DATASET.num_types(), restore=SHOULD_RESTORE,
                 model_name=MODEL_NAME, lr=LR, lr_decay=LR_DECAY,
                 batch_size=BATCH_SIZE, num_steps=NUM_STEPS,
                 embedding_size=EMBED_SIZE, rnn_layers=NUM_RNN_LAYERS,
                 max_grad_norm=MAX_GRAD_NORM, rnn_cell=RNN_CELL)
  
  with tf.Session(graph=GRAPH) as sess:
    m.start()
    #train(m, sess)
    hallucinate(m, sess)
