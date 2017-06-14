"""Testing setup for QALB."""

from __future__ import division, print_function

import os
import sys

import numpy as np
import tensorflow as tf

from ai.datasets import CharQALB
from ai.models import Seq2Seq


### HYPERPARAMETERS
tf.app.flags.DEFINE_float('lr', 5e-4, "Learning rate.")
tf.app.flags.DEFINE_float('lr_decay', 1., "Learning rate decay.")
tf.app.flags.DEFINE_integer('batch_size', 20, "Batch size.")
tf.app.flags.DEFINE_integer('embedding_size', 256, "Number of hidden units.")
tf.app.flags.DEFINE_integer('rnn_layers', 2, "Number of RNN layers.")
tf.app.flags.DEFINE_boolean('bidirectional_encoder', True, "Whether to use a"
                            " bidirectional RNN in the encoder's 1st layer.")
tf.app.flags.DEFINE_boolean('add_fw_bw', True, "Set to False to concatenate"
                            " and project the bidirectional RNN output.")
tf.app.flags.DEFINE_boolean('pyramid_encoder', False, "Set to True to use an"
                            " encoder that halves the time steps per layer.")
tf.app.flags.DEFINE_float('max_grad_norm', 5., "Clip gradients to this norm.")
tf.app.flags.DEFINE_float('epsilon', 1e-8, "Denominator constant for Adam.")
tf.app.flags.DEFINE_boolean('use_lstm', False, "Set to False to use GRUs.")
tf.app.flags.DEFINE_boolean('use_residual', False, "Set to True to add the RNN"
                            " inputs to the outputs.")
tf.app.flags.DEFINE_boolean('use_luong_attention', True, "Set to False to use"
                            " Bahdanau (additive) attention.")
tf.app.flags.DEFINE_integer('beam_size', 1, "Beam search size.")
tf.app.flags.DEFINE_integer('switch_to_sgd', None, "Set to a number of steps"
                            " to pass for the optimizer to switch to SGD.")
tf.app.flags.DEFINE_float('dropout', 1., "Keep probability for dropout on the"
                          "RNNs' non-recurrent connections.")
tf.app.flags.DEFINE_integer('gram_order', 1, "Size of the n-grams.")
tf.app.flags.DEFINE_float('p_sample_decay', 0., "Inverse sigmoid decay"
                          " parameter for scheduled sampling (0 = no sample).")

### CONFIG
tf.app.flags.DEFINE_integer('max_sentence_length', 400, "Max. word length of"
                            " training examples (both inputs and labels).")
tf.app.flags.DEFINE_integer('num_steps_per_eval', 10, "Number of steps to wait"
                            " before running the graph with the dev set.")
tf.app.flags.DEFINE_integer('num_steps_per_save', 50, "Number of steps to wait"
                            " before saving the trainable variables.")
tf.app.flags.DEFINE_string('decode', None, "Set to a path to run on a file.")
tf.app.flags.DEFINE_string('output_path', os.path.join('output', 'result.txt'),                         "Name of the output file with decoding results.")
tf.app.flags.DEFINE_boolean('restore', True, "Whether to restore the model.")
tf.app.flags.DEFINE_string('model_name', None, "Name of the output directory.")


FLAGS = tf.app.flags.FLAGS


def train():
  """Run a loop that continuously trains the model."""
  print("Building dynamic character-level QALB data...")
  dataset = CharQALB('QALB', batch_size=FLAGS.batch_size,
                     gram_order=FLAGS.gram_order,
                     max_input_length=FLAGS.max_sentence_length,
                     max_label_length=FLAGS.max_sentence_length)
  print("Building computational graph...")
  graph = tf.Graph()
  with graph.as_default():
    # pylint: disable=invalid-name
    m = Seq2Seq(num_types=dataset.num_types(),
                max_encoder_length=FLAGS.max_sentence_length,
                max_decoder_length=FLAGS.max_sentence_length,
                pad_id=dataset.type_to_ix['_PAD'],
                eos_id=dataset.type_to_ix['_EOS'],
                go_id=dataset.type_to_ix['_GO'], lr=FLAGS.lr,
                lr_decay=FLAGS.lr_decay, batch_size=FLAGS.batch_size,
                embedding_size=FLAGS.embedding_size,
                rnn_layers=FLAGS.rnn_layers,
                bidirectional_encoder=FLAGS.bidirectional_encoder,
                add_fw_bw=FLAGS.add_fw_bw,
                pyramid_encoder=FLAGS.pyramid_encoder,
                max_grad_norm=FLAGS.max_grad_norm, epsilon=FLAGS.epsilon,
                use_lstm=FLAGS.use_lstm, use_residual=FLAGS.use_residual,
                use_luong_attention=FLAGS.use_luong_attention, beam_size=1,
                dropout=FLAGS.dropout, restore=FLAGS.restore,
                model_name=FLAGS.model_name)

  with tf.Session(graph=graph) as sess:
    print("Initializing or restoring model...")
    m.start()
    print("Entering training loop...")
    while True:
      step = m.global_step.eval()
      
      # Gradient descent and backprop
      train_inputs, train_labels = dataset.get_batch()
      train_fd = {m.inputs: train_inputs, m.labels: train_labels}
      
      if FLAGS.switch_to_sgd and step >= FLAGS.switch_to_sgd:
        sess.run(m.sgd, feed_dict=train_fd)
      else:
        sess.run(m.adam, feed_dict=train_fd)
      
      # Decay sampling probability with inverse sigmoid decay
      k = FLAGS.p_sample_decay
      if k > 0:
        sess.run(tf.assign(m.p_sample, 1 - k / (k + np.exp(step/k))))
      
      if step % FLAGS.num_steps_per_eval == 0:
        
        # Show learning rate and sample outputs from training set
        lr, train_ppx, p_sample, train_summary = sess.run(
          [m.lr, m.perplexity, m.p_sample, m.summary_op], feed_dict=train_fd
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
        print(repr(dataset.untokenize(valid_inputs[0], join_str='')))
        print("Sample target:")
        print(repr(dataset.untokenize(valid_labels[0], join_str='')))
        print("Sample output:")
        print(repr(dataset.untokenize(valid_output[0], join_str='')))
        sys.stdout.flush()
      
      if step % FLAGS.num_steps_per_save == 0:
        print("==============================================================")
        print("Saving model...")
        m.save()
        print("Model saved. Resuming training...")
        sys.stdout.flush()


def decode():
  """Run a blind test on the file with path given by the `decode` flag."""
  print("Reading data...")
  with open(FLAGS.decode) as test_file:
    lines = test_file.readlines()
    max_length = max(map(lambda line: len(' '.join(line.split()[1:])), lines))
  
  print("Building dynamic word-level QALB data...")
  dataset = CharQALB('QALB', batch_size=1, gram_order=FLAGS.gram_order,
                     max_input_length=max_length,
                     max_label_length=max_length)
  print("Building computational graph...")
  graph = tf.Graph()
  with graph.as_default():
    # TODO: remove constants dependent on dataset instance; save them in model.
    # pylint: disable=invalid-name
    m = Seq2Seq(num_types=dataset.num_types(),
                max_encoder_length=max_length,
                max_decoder_length=max_length,
                pad_id=dataset.type_to_ix['_PAD'],
                eos_id=dataset.type_to_ix['_EOS'],
                go_id=dataset.type_to_ix['_GO'], batch_size=1,
                embedding_size=FLAGS.embedding_size,
                rnn_layers=FLAGS.rnn_layers,
                bidirectional_encoder=FLAGS.bidirectional_encoder,
                add_fw_bw=FLAGS.add_fw_bw,
                pyramid_encoder=FLAGS.pyramid_encoder, use_lstm=FLAGS.use_lstm,
                use_residual=FLAGS.use_residual,
                use_luong_attention=FLAGS.use_luong_attention,
                beam_size=FLAGS.beam_size, restore=True,
                model_name=FLAGS.model_name)
  
  with tf.Session(graph=graph) as sess:
    print("Restoring model...")
    m.start()
    print("Restored model (global step {})".format(m.global_step.eval()))
    
    with open(FLAGS.output_path, 'w') as output_file:
      for line in lines:
        line = ' '.join(line.split()[1:])
        print('Input:')
        print(line)
        ids = dataset.tokenize(line)
        while len(ids) < max_length:
          ids.append(dataset.type_to_ix['_PAD'])
        fd = {m.inputs: [ids], m.temperature: 1.}
        o_ids = sess.run(m.generative_output[0].sample_id, feed_dict=fd)[0]
        output = dataset.untokenize(o_ids, join_str='') + '\n'
        print('Output:')
        print(output)
        output_file.write(output)

def main(_):
  """Called by `tf.app.run` method."""
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
