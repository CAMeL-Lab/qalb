from __future__ import division
from __future__ import print_function

from collections import deque

import tensorflow as tf

from ai.datasets import TextFile
from ai.models import NextGram


# HYPERPARAMETERS
LR = 1.
LR_DECAY = .5
NUM_BAD = 20
BATCH_SIZE = 20
NUM_STEPS = 40
EMBED_SIZE = 100
NUM_RNN_LAYERS = 3
MAX_GRAD_NORM = 5.
RNN_CELL = tf.contrib.rnn.LSTMBlockCell
GRAM_ORDER = 2
SHUFFLE_DATA = True

# Other config
steps_per_eval = 20
steps_per_save = 500
should_restore = False
model_name = 'bigram-bible-3l'


# Read and preprocess data
print("Reading data...")
dataset = TextFile('bible.txt', batch_size=BATCH_SIZE, gram_order=GRAM_ORDER,
  shuffle=SHUFFLE_DATA, num_steps=NUM_STEPS)

# Build the computational graph
graph = tf.Graph()
with graph.as_default():
  m = NextGram(alphabet_size=dataset.num_types(), restore=should_restore,
    model_name=model_name, lr=LR, lr_decay=LR_DECAY, batch_size=BATCH_SIZE,
    num_steps=NUM_STEPS, embed_size=EMBED_SIZE, num_rnn_layers=NUM_RNN_LAYERS,
    max_grad_norm=MAX_GRAD_NORM, rnn_cell=RNN_CELL)


def train():
  with tf.Session(graph=graph) as sess:
    m.start()
    # Recent improvements tracked for learning rate decay
    num_passed = 0
    average = 0
    last_perplexities = deque(NUM_BAD * [0.], NUM_BAD)
    # Training loop
    print("Entering training loop (step {})".format(m.global_step.eval()))
    while True:
      for train_batch, valid_batch in dataset.get_batches():
        # Construct feed dicts
        train_inputs, train_labels, _ = zip(*train_batch)
        train_fd = {m.inputs: train_inputs, m.labels: train_labels}
        valid_inputs, valid_labels, _ = zip(*valid_batch)
        valid_fd = {m.inputs: valid_inputs, m.labels: valid_labels}
        # OPTIONAL: stop after the specified number of steps
        step = m.global_step.eval()
        if step > 5000:
          exit()
        # Training step
        sess.run(m.train_op, feed_dict=train_fd)
        # Evaluation every specified number of steps
        if step % steps_per_eval == 0:
          # Save train stats
          lr, train_perp, train_summary = sess.run(
            [m._lr, m.perplexity, m.summary_op], feed_dict=train_fd)
          m.train_writer.add_summary(train_summary, global_step=step)
          # Save valid stats, and generate some sample output
          output, valid_perp, valid_summary = sess.run(
            [m.output, m.perplexity, m.summary_op], feed_dict=valid_fd)
          m.valid_writer.add_summary(valid_summary, global_step=step)
          # Print generated samples
          print("=========================================================")
          print("Step {0} (lr={1}, train_perp={2}, valid_perp={3})".format(
            step, str(lr)[:6], str(train_perp)[:6], str(valid_perp)[:6]))
          print("=========================================================")
          print("Sample output:")
          print(dataset.untokenize([valid_inputs[0][0]]) + \
                dataset.untokenize(output))
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
        if step % steps_per_save == 0:
          m.save()

def generate():
  with tf.Session(graph=graph) as sess:
    m.start()
    while True:
      seed = dataset.tokenize(raw_input('Seed: '))
      prediction_length = int(raw_input('Prediction length: '))
      # Use generative model
      embeddings = sess.run(m.get_embeddings, feed_dict={m.tokens: seed})
      predictions = [sess.run(tf.tile(tf.reshape(embeddings,
        [1, 1, m.embed_size]), [m.batch_size, 1, 1]))]
      state = sess.run(tf.stack(m.initial_state, axis=0))
      for i in xrange(prediction_length):
        gram = None
        if i < len(seed):
          gram = seed[i]
        else:
          gram = predictions[-1]
        #_input = sess.run(tf.)
        output, state = sess.run(m.gen_output,
          feed_dict={m.seed: gram, m.gen_state: state})
        predictions.append(output[0])
        print(dataset.untokenize(predictions))


if __name__ == '__main__':
  train()
  #generate()
