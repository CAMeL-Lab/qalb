"""Testing setup for QALB shared task."""

from __future__ import division, print_function

import io
import os
import sys
import timeit

from six.moves import xrange
import numpy as np
import tensorflow as tf

from ai.datasets import CharQALB
from ai.models import Seq2Seq


### HYPERPARAMETERS
tf.app.flags.DEFINE_float('adam_lr', 5e-4, "Adam learning rate.")
tf.app.flags.DEFINE_float('adam_lr_decay', 1., "Adam learning rate decay.")
tf.app.flags.DEFINE_float('gd_lr', .1, "Gradient descent learning rate.")
tf.app.flags.DEFINE_float('gd_lr_decay', 1., "Gradient descent learning rate"
                          " decay.")
tf.app.flags.DEFINE_integer('batch_size', 20, "Batch size.")
tf.app.flags.DEFINE_integer('embedding_size', 128, "Number of hidden units.")
tf.app.flags.DEFINE_integer('rnn_layers', 2, "Number of RNN layers.")
tf.app.flags.DEFINE_boolean('bidirectional_encoder', True, "Whether to use a"
                            " bidirectional RNN in the encoder's 1st layer.")
tf.app.flags.DEFINE_string('bidirectional_mode', 'add', "Set to 'add',"
                           " 'concat' or 'project'.")
tf.app.flags.DEFINE_boolean('pyramid_encoder', False, "Set to True to use an"
                            " encoder that halves the time steps per layer.")
tf.app.flags.DEFINE_boolean('use_lstm', False, "Set to False to use GRUs.")
tf.app.flags.DEFINE_boolean('use_residual', False, "Set to True to add the RNN"
                            " inputs to the outputs.")
tf.app.flags.DEFINE_string('attention', None, "'bahdanau' or 'luong' (none"
                           " by default).")
tf.app.flags.DEFINE_float('dropout', 1., "Keep probability for dropout on the"
                          "RNNs' non-recurrent connections.")
tf.app.flags.DEFINE_float('max_grad_norm', 5., "Clip gradients to this norm.")
tf.app.flags.DEFINE_float('epsilon', 1e-8, "Denominator constant for Adam.")
tf.app.flags.DEFINE_integer('beam_size', 1, "Beam search size.")
tf.app.flags.DEFINE_float('p_sample_decay', 0., "Inverse sigmoid decay"
                          " parameter for scheduled sampling (0 = no sample).")
tf.app.flags.DEFINE_integer('switch_to_sgd', None, "Set to a number of steps"
                            " to pass for the optimizer to switch to SGD.")

### CONFIG
tf.app.flags.DEFINE_integer('max_sentence_length', 400, "Max. word length of"
                            " training examples (both inputs and labels).")
tf.app.flags.DEFINE_integer('num_steps_per_eval', 20, "Number of steps to wait"
                            " before running the graph with the dev set.")
tf.app.flags.DEFINE_integer('num_steps_per_save', 100, "Number of steps"
                            " before saving the trainable variables.")
tf.app.flags.DEFINE_string('extension', '', "Extensions of data files.")
tf.app.flags.DEFINE_string('decode', None, "Set to a path to run on a file.")
tf.app.flags.DEFINE_string('output_path', os.path.join('output', 'result.txt'),
                           "Name of the output file with decoding results.")
tf.app.flags.DEFINE_boolean('restore', True, "Whether to restore the model.")
tf.app.flags.DEFINE_string('model_name', None, "Name of the output directory.")


FLAGS = tf.app.flags.FLAGS


def train():
  """Run a loop that continuously trains the model."""
  
  print("Building dynamic character-level QALB data...")
  dataset = CharQALB(
    'QALB', extension=FLAGS.extension,
    max_input_length=FLAGS.max_sentence_length,
    max_label_length=FLAGS.max_sentence_length)
  
  print("Building computational graph...")
  graph = tf.Graph()
  with graph.as_default():
    # pylint: disable=invalid-name
    m = Seq2Seq(
      num_types=dataset.num_types(),
      max_encoder_length=FLAGS.max_sentence_length,
      max_decoder_length=FLAGS.max_sentence_length,
      pad_id=dataset.type_to_ix['_PAD'],
      eos_id=dataset.type_to_ix['_EOS'],
      go_id=dataset.type_to_ix['_GO'],
      adam_lr=FLAGS.adam_lr, adam_lr_decay=FLAGS.adam_lr_decay,
      gd_lr=FLAGS.gd_lr, gd_lr_decay=FLAGS.gd_lr_decay,
      batch_size=FLAGS.batch_size, embedding_size=FLAGS.embedding_size,
      rnn_layers=FLAGS.rnn_layers,
      bidirectional_encoder=FLAGS.bidirectional_encoder,
      bidirectional_mode=FLAGS.bidirectional_mode,
      pyramid_encoder=FLAGS.pyramid_encoder, max_grad_norm=FLAGS.max_grad_norm,
      epsilon=FLAGS.epsilon, use_lstm=FLAGS.use_lstm,
      use_residual=FLAGS.use_residual, attention=FLAGS.attention, beam_size=1,
      dropout=FLAGS.dropout, restore=FLAGS.restore,
      model_name=FLAGS.model_name)
  
  with tf.Session(graph=graph) as sess:
    print("Initializing or restoring model...")
    m.start()
    print("Entering training loop...")
    
    while True:
      step = m.global_step.eval()
      
      # Gradient descent and backprop
      # TODO: add lr decays
      train_inputs, train_labels = dataset.get_batch(FLAGS.batch_size)
      train_fd = {m.inputs: train_inputs, m.labels: train_labels}
      
      # Wrap into function to measure running time
      def train_step():
        if FLAGS.switch_to_sgd and step >= FLAGS.switch_to_sgd:
          sess.run(m.sgd, feed_dict=train_fd)
        else:
          sess.run(m.adam, feed_dict=train_fd)
      
      print("Global step {0} ({1}s)".format(
        step, timeit.timeit(train_step, number=1)))
      
      # Decay sampling probability with inverse sigmoid decay
      # TODO: add options for different decays
      k = FLAGS.p_sample_decay
      if k > 0:
        sess.run(tf.assign(m.p_sample, 1 - k / (k + np.exp(step/k))))
      
      if step % FLAGS.num_steps_per_eval == 0:
        valid_inputs, valid_labels = dataset.get_batch(
          FLAGS.batch_size, draw_from_valid=True)
        valid_fd = {m.inputs: valid_inputs, m.labels: valid_labels}
        
        # Run training and validation perplexity and samples
        
        lr, train_ppx, train_output, p_sample, train_ppx_summ = sess.run([
          m.adam_lr,
          m.perplexity,
          m.output,
          m.p_sample,
          m.perplexity_summary,
        ], feed_dict=train_fd)
        
        valid_ppx, valid_output, infer_output, valid_ppx_summ = sess.run([
          m.perplexity,
          m.output,
          m.generative_output,
          m.perplexity_summary,
        ], feed_dict=valid_fd)
        infer_output = infer_output[0].sample_id
        
        # Run training, validation and inference Levenshtein distances
        
        train_lev, train_lev_summ = sess.run(
          [m.lev_out, m.lev_summary],
          feed_dict={m.lev_in: train_output, m.labels: train_labels})
        
        valid_lev, valid_lev_summ = sess.run(
          [m.lev_out, m.lev_summary],
          feed_dict={m.lev_in: valid_output, m.labels: valid_labels})
        
        infer_lev, infer_lev_summ = sess.run(
          [m.lev_out, m.lev_summary],
          feed_dict={m.lev_in: infer_output, m.labels: valid_labels})
        
        # Write summaries to TensorBoard
        m.train_writer.add_summary(train_ppx_summ, global_step=step)
        m.train_writer.add_summary(train_lev_summ, global_step=step)
        m.valid_writer.add_summary(valid_ppx_summ, global_step=step)
        m.valid_writer.add_summary(valid_lev_summ, global_step=step)
        m.infer_writer.add_summary(infer_lev_summ, global_step=step)
        
        # Display results to stdout
        print("  lr:", lr)
        print("  p_sample:", p_sample)
        print("  train_ppx:", train_ppx)
        print("  train_lev:", train_lev)
        print("  valid_ppx:", valid_ppx)
        print("  valid_lev:", valid_lev)
        print("  infer_lev:", infer_lev)
        print("Input:")
        print(dataset.untokenize(valid_inputs[0], join_str=''))
        print("Target:")
        print(dataset.untokenize(valid_labels[0], join_str=''))
        print("Output with ground truth:")
        print(dataset.untokenize(valid_output[0], join_str=''))
        print("Decoded output:")
        print(dataset.untokenize(infer_output[0], join_str=''))
        sys.stdout.flush()
      
      if step % FLAGS.num_steps_per_save == 0:
        print("Saving model...")
        m.save()
        print("Model saved. Resuming training...")
        sys.stdout.flush()


def decode():
  """Run a blind test on the file with path given by the `decode` flag."""
  
  print("Reading data...")
  with io.open(FLAGS.decode, encoding='utf-8') as test_file:
    lines = test_file.readlines()
    # Get the largest sentence length to set an upper bound to the decoder
    # TODO: add some heuristic to allow that to increase a bit more
    max_length = max([len(line) for line in lines])
    
  print("Building dynamic word-level QALB data...")
  dataset = CharQALB(
    'QALB', extension=FLAGS.extension,
    max_input_length=max_length, max_label_length=max_length)
  
  print("Building computational graph...")
  graph = tf.Graph()
  with graph.as_default():
    # TODO: remove constants dependent on dataset instance; save them in model.
    # pylint: disable=invalid-name
    m = Seq2Seq(
      num_types=dataset.num_types(),
      max_encoder_length=max_length, max_decoder_length=max_length,
      pad_id=dataset.type_to_ix['_PAD'],
      eos_id=dataset.type_to_ix['_EOS'],
      go_id=dataset.type_to_ix['_GO'],
      batch_size=1, embedding_size=FLAGS.embedding_size,
      rnn_layers=FLAGS.rnn_layers,
      bidirectional_encoder=FLAGS.bidirectional_encoder,
      bidirectional_mode=FLAGS.bidirectional_mode,
      pyramid_encoder=FLAGS.pyramid_encoder, use_lstm=FLAGS.use_lstm,
      use_residual=FLAGS.use_residual, attention=FLAGS.attention,
      beam_size=FLAGS.beam_size, restore=True, model_name=FLAGS.model_name)
  
  with tf.Session(graph=graph) as sess:
    print("Restoring model...")
    m.start()
    print("Restored model (global step {})".format(m.global_step.eval()))
    
    with io.open(FLAGS.output_path, 'w', encoding='utf-8') as output_file:
      for line in lines:
        print('Input:')
        print(line)
        ids = dataset.tokenize(line)
        while len(ids) < max_length:
          ids.append(dataset.type_to_ix['_PAD'])
        feed_dict = {m.inputs: [ids], m.temperature: 1.}
        o_ids = sess.run(m.generative_output[0].sample_id, feed_dict=feed_dict)
        # Remove the _EOS token
        output = dataset.untokenize(o_ids[0], join_str='')[:-1] + '\n'
        print('Output:')
        print(output)
        output_file.write(output)


def main(_):
  """Called by `tf.app.run` method."""
  if not FLAGS.model_name:
    raise ValueError(
      "Undefined model name. Perhaps you forgot to set the --model_name flag?")
  
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == '__main__':
  tf.app.run()
