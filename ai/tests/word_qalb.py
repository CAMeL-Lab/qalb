"""Testing setup for QALB."""

from __future__ import division, print_function

from collections import deque

import numpy as np
import tensorflow as tf

from ai.datasets import qalb
from ai.models import Seq2Seq


class DynamicWordQALB(qalb.DynamicQALB, qalb.WordQALB):
  pass


### HYPERPARAMETERS
tf.app.flags.DEFINE_float('lr', 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float('lr_decay', .9, "Learning rate decay.")
tf.app.flags.DEFINE_integer('lr_decay_threshold', 5, "Number of steps"
                            " with no improvement for learning rate decay.")
tf.app.flags.DEFINE_integer('batch_size', 20, "Batch size.")
tf.app.flags.DEFINE_integer('embedding_size', 256, "Number of hidden units.")
tf.app.flags.DEFINE_integer('max_words', 20000, "Max. number of unique words.")
tf.app.flags.DEFINE_integer('max_sentence_length', 50, "Max. word length of"
                            " training examples (both inputs and labels).")
tf.app.flags.DEFINE_integer('rnn_layers', 1, "Number of RNN layers.")
tf.app.flags.DEFINE_boolean('bidirectional_encoder', True, "Whether to use a"
                            " bidirectional RNN in the encoder.")
tf.app.flags.DEFINE_float('max_grad_norm', 5., "Clip gradients to this norm.")
tf.app.flags.DEFINE_boolean('use_lstm', False, "Set to False to use GRUs.")
tf.app.flags.DEFINE_boolean('use_luong_attention', True, "Set to False to use"
                            " Bahdanau (additive) attention.")
tf.app.flags.DEFINE_float('initial_p_sample', 0., "Initial probability to."
                          "sample from the decoder's own predictions.")

### CONFIG
tf.app.flags.DEFINE_integer('num_steps_per_eval', 10, "Number of steps to wait"
                            " before running the graph with the dev set.")
tf.app.flags.DEFINE_integer('num_steps_per_save', 50, "Number of steps to wait"
                            " before saving the trainable variables.")
tf.app.flags.DEFINE_integer('stop_threshold', 50, "Number of steps with no"
                            " improvement for early termination.")
tf.app.flags.DEFINE_string('decode', None, "Set to a path to run on a file.")
tf.app.flags.DEFINE_boolean('restore', True, "Whether to restore the model.")
tf.app.flags.DEFINE_string('model_name', None, "Name of the output directory.")


FLAGS = tf.app.flags.FLAGS


def train():
  """Run a loop that continuously trains the model."""
  print("Building dynamic word-level QALB data...")
  dataset = DynamicWordQALB('QALB',
                            max_types=FLAGS.max_words,
                            batch_size=FLAGS.batch_size,
                            max_input_length=FLAGS.max_sentence_length,
                            max_label_length=FLAGS.max_sentence_length)
  print("Building computational graph...")
  graph = tf.Graph()
  with graph.as_default():
    # TODO: move this to the model
    if FLAGS.use_lstm:
      RNN_CELL = tf.contrib.rnn.LSTMBlockCell 
    else:
      RNN_CELL = tf.contrib.rnn.GRUBlockCell
    m = Seq2Seq(num_types=dataset.num_types(),
                max_encoder_length=FLAGS.max_sentence_length,
                max_decoder_length=FLAGS.max_sentence_length,
                pad_id=dataset.type_to_ix['_PAD'],
                eos_id=dataset.type_to_ix['_EOS'],
                go_id=dataset.type_to_ix['_GO'],
                lr=FLAGS.lr, lr_decay=FLAGS.lr_decay,
                batch_size=FLAGS.batch_size, embedding_size=FLAGS.embedding_size,
                rnn_layers=FLAGS.rnn_layers,
                bidirectional_encoder=FLAGS.bidirectional_encoder,
                max_grad_norm=FLAGS.max_grad_norm, rnn_cell=RNN_CELL,
                use_luong_attention=FLAGS.use_luong_attention,
                initial_p_sample=FLAGS.initial_p_sample,
                restore=FLAGS.restore, model_name=FLAGS.model_name)

  with tf.Session(graph=graph) as sess:
    print("Initializing or restoring model...")
    m.start()
    
    # Recent improvements tracked for learning rate decay
    num_passed_lr = 0
    average_lr = 0
    last_perplexities_lr = deque(
      FLAGS.lr_decay_threshold * [0.], FLAGS.lr_decay_threshold
    )
    # Recent improvements tracked for early termination
    num_passed_stop = 0
    average_stop = 0
    last_perplexities_stop = deque(
      FLAGS.stop_threshold * [0.], FLAGS.stop_threshold
    )
    
    print("Entering training loop...")
    while True:
      # Gradient descent and backprop
      train_inputs, train_labels = dataset.get_batch()
      train_fd = {m.inputs: train_inputs, m.labels: train_labels}
      sess.run(m.train_op, feed_dict=train_fd)
      
      # Linear model for change in the sampling probability
      step = m.global_step.eval()
      # sess.run(tf.assign(m._p_sample,
      #   tf.reduce_min([1., tf.cast(step, tf.float32) * P_SAMPLE_M]))
      # )
      
      if step % FLAGS.num_steps_per_eval == 0:
        # Show learning rate and sample outputs from training set
        lr, train_ppx, p_sample, train_summary = sess.run(
          [m._lr, m.perplexity, m._p_sample, m.summary_op], feed_dict=train_fd
        )
        # Evaluate and show samples on validation set
        valid_inputs, valid_labels = dataset.get_batch(draw_from_valid=True)
        valid_fd = {m.inputs: valid_inputs, m.labels: valid_labels}
        valid_ppx, valid_output, valid_summary = sess.run(
          [m.perplexity, m.output, m.summary_op], feed_dict=valid_fd
        )
        # Write summaries to TensorBoard
        m.train_writer.add_summary(train_summary, global_step=step)
        m.valid_writer.add_summary(valid_summary, global_step=step)
        print("==============================================================")
        print("Step {0} (lr={1}, p_sample={2}, train_ppx={3}, valid_ppx={4})"
              "".format(step, lr, p_sample, train_ppx, valid_ppx)
        )
        print("==============================================================")
        print("Sample input:")
        print(dataset.untokenize(valid_inputs[0]))
        print("Sample target:")
        print(dataset.untokenize(valid_labels[0]))
        print("Sample output:")
        print(dataset.untokenize(valid_output[0]))
        
        # Decay the learning rate if there has been no recent improvement
        last_perplexities_lr.appendleft(train_ppx)
        new_average = sum(last_perplexities_lr) / FLAGS.lr_decay_threshold
        if num_passed_lr == FLAGS.lr_decay_threshold:
          if new_average > average_lr:
            num_passed_lr = 0
            sess.run(m.decay_lr)
        else:
          num_passed_lr += 1
        average_lr = new_average
        
        # Stop if there has been no improvement for a while
        last_perplexities_stop.appendleft(valid_ppx)
        new_average = sum(last_perplexities_stop) / FLAGS.stop_threshold
        if num_passed_stop == FLAGS.stop_threshold:
          if new_average > average_stop:
            num_passed_stop = 0
            exit()
        else:
          num_passed_stop += 1
        average_stop = new_average
      
      if step % FLAGS.num_steps_per_save == 0:
        print("==============================================================")
        print("Saving model...")
        m.save()
        print("Model saved. Resuming training...")


def decode():
  print("Building dynamic word-level QALB data...")
  dataset = DynamicWordQALB('QALB',
                            max_types=FLAGS.max_words,
                            batch_size=1,
                            max_input_length=FLAGS.max_sentence_length,
                            max_label_length=FLAGS.max_sentence_length)
  print("Building computational graph...")
  graph = tf.Graph()
  with graph.as_default():
    if FLAGS.use_lstm:
      RNN_CELL = tf.contrib.rnn.LSTMBlockCell 
    else:
      RNN_CELL = tf.contrib.rnn.GRUBlockCell
    m = Seq2Seq(num_types=dataset.num_types(),
                max_encoder_length=None,
                max_decoder_length=FLAGS.max_sentence_length,
                pad_id=dataset.type_to_ix['_PAD'],
                eos_id=dataset.type_to_ix['_EOS'],
                go_id=dataset.type_to_ix['_GO'],
                batch_size=1, embedding_size=FLAGS.embedding_size,
                rnn_layers=FLAGS.rnn_layers,
                bidirectional_encoder=FLAGS.bidirectional_encoder,
                rnn_cell=RNN_CELL,
                use_luong_attention=FLAGS.use_luong_attention,
                restore=True, model_name=FLAGS.model_name)
  
  with tf.Session(graph=graph) as sess:
    print("Restoring model...")
    m.start()
    print("Restored model (global step {})".format(m.global_step.eval()))
    
    with open(FLAGS.decode) as f:
      lines = f.readlines()
    for line in lines:
      ids = [dataset.tokenize(line.split()[1:])]
      decoded = sess.run(m.generative_output, feed_dict={m.inputs: ids})
      print(dataset.untokenize(decoded[0]))

def main(_):
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
