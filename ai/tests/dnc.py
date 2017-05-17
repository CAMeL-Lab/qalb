import tensorflow as tf

from ai.models import NextGramDNC
from ai.datasets import TextFile


# HYPERPARAMETERS
LR = 1.
TIME_STEPS = 20
BATCH_SIZE = 1
GRAM_ORDER = 2
SHUFFLE_DATA = True


# Other config
model_name = 'bigram-bible-dnc'
should_restore = False


# Read and preprocess data
print("Reading data...")
dataset = TextFile('ai/datasets/data/bible.txt', batch_size=BATCH_SIZE,
  gram_order=GRAM_ORDER, shuffle=SHUFFLE_DATA, num_steps=TIME_STEPS)

graph = tf.Graph()
with graph.as_default():
  m = NextGramDNC(dataset.size(),restore=should_restore, model_name=model_name,
    lr=LR)

with tf.Session(graph=graph) as sess:
  m.start()
