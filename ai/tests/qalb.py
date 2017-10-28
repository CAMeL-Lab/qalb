"""Testing setup for QALB shared task."""

import io
import os
import sys
import timeit

import numpy as np
import tensorflow as tf
import editdistance

from ai.datasets import QALB
from ai.models import Seq2Seq


### HYPERPARAMETERS
tf.app.flags.DEFINE_float('lr', 5e-4, "Initial learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 64, "Batch size.")
tf.app.flags.DEFINE_integer('embedding_size', 64, "Embedding dimensionality.")
tf.app.flags.DEFINE_integer('hidden_size', 256, "Number of hidden units.")
tf.app.flags.DEFINE_integer('rnn_layers', 2, "Number of RNN layers.")
tf.app.flags.DEFINE_boolean('bidirectional_encoder', True, "Whether to use a"
                            " bidirectional RNN in the encoder's 1st layer.")
tf.app.flags.DEFINE_string('bidirectional_mode', 'add', "Set to 'add',"
                           " 'concat' or 'project'.")
tf.app.flags.DEFINE_boolean('use_lstm', False, "Set to False to use GRUs.")
tf.app.flags.DEFINE_boolean('use_residual', False, "Set to True to add the RNN"
                            " inputs to the outputs.")
tf.app.flags.DEFINE_string('attention', 'luong', "'bahdanau' or 'luong'"
                           " (default is 'luong').")
tf.app.flags.DEFINE_float('dropout', 1., "Keep probability for dropout on the"
                          "RNNs' non-recurrent connections.")
tf.app.flags.DEFINE_float('max_grad_norm', 5., "Clip gradients to this norm.")
tf.app.flags.DEFINE_float('epsilon', 1e-8, "Denominator constant for Adam.")
tf.app.flags.DEFINE_integer('beam_size', 3, "Beam search size.")
# Set this to > 0 even if no decay
tf.app.flags.DEFINE_float('p_sample', 0, 'Initial sampling probability.')
tf.app.flags.DEFINE_integer('switch_to_sgd', None, "Set to a number of steps"
                            " to pass for the optimizer to switch to SGD.")

### CONFIG
tf.app.flags.DEFINE_integer('max_sentence_length', 400, "Max. word length of"
                            " training examples (both inputs and labels).")
tf.app.flags.DEFINE_integer('num_steps_per_eval', 50, "Number of steps to wait"
                            " before running the graph with the dev set.")
tf.app.flags.DEFINE_integer('num_steps_per_save', 100, "Number of steps"
                            " before saving the trainable variables.")
tf.app.flags.DEFINE_integer('max_epochs', 20, "Number of epochs to run"
                            " (0 = no limit).")
tf.app.flags.DEFINE_string('extension', '', "Extensions of data files.")
tf.app.flags.DEFINE_string('decode', None, "Set to a path to run on a file.")
tf.app.flags.DEFINE_string('output_path', os.path.join('output', 'result.txt'),
                           "Name of the output file with decoding results.")
tf.app.flags.DEFINE_boolean('restore', True, "Whether to restore the model.")
tf.app.flags.DEFINE_string('model_name', None, "Name of the output directory.")


FLAGS = tf.app.flags.FLAGS


def untokenize_batch(dataset, id_batch):
  """Return the UTF-8 sequences of the given batch of ids."""
  return [dataset.untokenize(dataset.clean(s)) for s in id_batch]


def levenshtein(proposed, gold):
  """Return the normalized Levenshtein distance of the given strings."""
  lev_densities = []
  for x, y in zip(proposed, gold):
    lev_densities.append(editdistance.eval(x, y) / len(y))
  return sum(lev_densities) / len(lev_densities)


def train():
  """Run a loop that continuously trains the model."""
  
  print("Building dynamic character-level QALB data...")
  dataset = QALB(
    'QALB', extension=FLAGS.extension, shuffle=True,
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
      batch_size=FLAGS.batch_size, embedding_size=FLAGS.embedding_size,
      hidden_size=FLAGS.hidden_size, rnn_layers=FLAGS.rnn_layers,
      bidirectional_encoder=FLAGS.bidirectional_encoder,
      bidirectional_mode=FLAGS.bidirectional_mode,
      use_lstm=FLAGS.use_lstm, use_residual=FLAGS.use_residual,
      attention=FLAGS.attention, dropout=FLAGS.dropout,
      max_grad_norm=FLAGS.max_grad_norm, epsilon=FLAGS.epsilon, 
      beam_size=FLAGS.beam_size, restore=FLAGS.restore,
      model_name=FLAGS.model_name)
  
  with tf.Session(graph=graph) as sess:
    print("Initializing or restoring model...")
    m.start()
    
    # If the model was not restored, set variable hyperparameters to their
    # provided initial values
    if sess.run(m.lr) == 0:
      sess.run(tf.assign(m.lr, FLAGS.lr))
    if sess.run(m.p_sample) == 0:
      sess.run(tf.assign(m.p_sample, FLAGS.p_sample))
    
    # Get the number of epochs that have passed (easier by getting batches now)
    step = m.global_step.eval()
    batches = dataset.get_train_batches(m.batch_size)
    epoch = step // len(batches)
    
    while not FLAGS.max_epochs or epoch <= FLAGS.max_epochs:
      print("=====EPOCH {}=====".format(epoch))
      sys.stdout.flush()
      while step < (epoch + 1) * len(batches):
        step = m.global_step.eval()
        
        # Gradient descent and backprop
        # TODO: add lr decays
        train_inputs, train_labels = zip(*batches[step % len(batches)])
        train_fd = {m.inputs: train_inputs, m.labels: train_labels}
        
        # Wrap into function to measure running time
        def train_step():
          if FLAGS.switch_to_sgd and step >= FLAGS.switch_to_sgd:
            sess.run(m.sgd, feed_dict=train_fd)
          else:
            sess.run(m.adam, feed_dict=train_fd)
        
        print("Global step {0} ({1}s)".format(
          step, timeit.timeit(train_step, number=1)))
        
        if step % FLAGS.num_steps_per_eval == 0:
          valid_inputs, valid_labels = dataset.get_valid_batch(m.batch_size)
          valid_fd = {m.inputs: valid_inputs, m.labels: valid_labels}
          
          # Run training and validation perplexity and samples
          
          lr, train_ppx, train_output, p_sample, train_ppx_summ = sess.run([
            m.lr,
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
          
          # Convert data to UTF-8 strings for evaluation and display
          train_labels = untokenize_batch(dataset, train_labels)
          train_output = untokenize_batch(dataset, train_output)
          valid_inputs = untokenize_batch(dataset, valid_inputs)
          valid_labels = untokenize_batch(dataset, valid_labels)
          valid_output = untokenize_batch(dataset, valid_output)
          infer_output = untokenize_batch(dataset, infer_output)
          
          # Run training, validation and inference Levenshtein distances
          train_lev = levenshtein(train_output, train_labels)
          valid_lev = levenshtein(valid_output, valid_labels)
          infer_lev = levenshtein(infer_output, valid_labels)
          
          train_lev_summ = sess.run(
            m.lev_summary, feed_dict={m.lev: train_lev})
          valid_lev_summ = sess.run(
            m.lev_summary, feed_dict={m.lev: valid_lev})
          infer_lev_summ = sess.run(
            m.infer_lev_summary, feed_dict={m.infer_lev: infer_lev})
          
          # Write summaries to TensorBoard
          m.train_writer.add_summary(train_ppx_summ, global_step=step)
          m.train_writer.add_summary(train_lev_summ, global_step=step)
          m.valid_writer.add_summary(valid_ppx_summ, global_step=step)
          m.valid_writer.add_summary(valid_lev_summ, global_step=step)
          m.valid_writer.add_summary(infer_lev_summ, global_step=step)
          
          # Display results to stdout
          print("  lr:", lr)
          print("  p_sample:", p_sample)
          print("  train_ppx:", train_ppx)
          print("  train_lev:", train_lev)
          print("  valid_ppx:", valid_ppx)
          print("  valid_lev:", valid_lev)
          print("  infer_lev:", infer_lev)
          print("Input:")
          print(valid_inputs[0])
          print("Target:")
          print(valid_labels[0])
          print("Output with ground truth:")
          print(valid_output[0])
          print("Decoded output:")
          print(infer_output[0])
        
        if step % FLAGS.num_steps_per_save == 0:
          print("Saving model...")
          m.save()
          print("Model saved. Resuming training...")
        
        sys.stdout.flush()
      
      # Epoch about to be done - reshuffle the data and get new batches
      batches = dataset.get_train_batches(m.batch_size)
      epoch += 1


def decode():
  """Run a blind test on the file with path given by the `decode` flag."""
  
  print("Reading data...")
  with io.open(FLAGS.decode, encoding='utf-8') as test_file:
    lines = test_file.readlines()
    # Get the largest sentence length to set an upper bound to the decoder
    # TODO: add some heuristic to allow that to increase a bit more
    max_length = max([len(line) for line in lines])
    
  print("Building dynamic word-level QALB data...")
  dataset = QALB(
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
      hidden_size=FLAGS.hidden_size, rnn_layers=FLAGS.rnn_layers,
      bidirectional_encoder=FLAGS.bidirectional_encoder,
      bidirectional_mode=FLAGS.bidirectional_mode,
      use_lstm=FLAGS.use_lstm, use_residual=FLAGS.use_residual,
      attention=FLAGS.attention, beam_size=FLAGS.beam_size,
      restore=True, model_name=FLAGS.model_name)
  
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
        output = sess.run(m.generative_output, feed_dict=feed_dict)
        output = untokenize_batch(dataset, output)[0] + '\n'
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
