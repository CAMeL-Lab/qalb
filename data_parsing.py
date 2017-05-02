from six.moves import xrange

import json
import random
import re


class Dataset(object):
  """Base class for parsing character-level text datasets. Includes defaults
     for special tokens, tokenizing and untokenizing methods, dataset
     preparation, and batch generation."""
  
  def __init__(self, batch_size=20, num_steps=50, gram_order=1, shuffle=False):
    """Keyword arguments:
       `batch_size`: number of lines per training batch,
       `num_steps`: specifies the max length of a line,
       `gram_order`: number of characters in the n-gram,
       `shuffle`: whether to shuffle the list of generated triples."""
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.gram_order = gram_order
    self.shuffle = shuffle
    
    # Data structures to tokenize and untokenize data in O(1) time
    self.ix_to_char = ['_PAD', '_EOS']
    # The `char_to_ix` dict will be {'_PAD': 0, '_EOS': 1} but the code below
    # allows to add any extra defaults without modifying both data structures.
    dictarg = lambda i: [self.ix_to_char[i], i]
    self.char_to_ix = dict(map(dictarg, xrange(len(self.ix_to_char))))
  
  def tokenize(self, input_str, add_eos=False):
    """Converts the argument string `s` to a list of integer tokens, each
       represents a unique character. If the charachter is not registered, it
       will be added to the `char_to_ix` and `ix_to_char` attributes."""
    result = []
    for i in xrange(len(input_str) - (self.gram_order - 1)):
      s = input_str[i:i+self.gram_order]
      if not (s in self.char_to_ix):
        self.char_to_ix[s] = len(self.ix_to_char)
        self.ix_to_char.append(s)
      result.append(self.char_to_ix[s])
    if add_eos:
      result.append(self.char_to_ix['_EOS'])
    return result
  
  def untokenize(self, tokens):
    """Converts the argument list of integer ids back to a string."""
    result = ''
    for t in tokens:
      # TODO: find a way to include the special tokens, even if more are added.
      if t > 1:
        char = self.ix_to_char[t][0]
        result += char
    return result
  
  # TODO: allow any number disjoint subsets of the triples and custom ratios.
  # By default 70% of the data is for training and 30% for validation.
  def make_triples(self, lines):
    """Given a list `lines` containing lists of ids, creates triples in form
      (line, shifted_line, next_line)."""
    n = len(lines) - 1
    make_triple = lambda i: (lines[i][:-1], lines[i][1:], lines[i+1][:-1])
    triples = map(make_triple, xrange(n))
    self.train_triples = triples[:int(n * .7)]
    self.valid_triples = triples[int(n * .7):]
  
  def get_batches(self):
    """Groups the triples into batches, and allows randomized order."""
    if self.shuffle:
      random.shuffle(self.train_triples)
      random.shuffle(self.valid_triples)
    # Put the triples into batches.
    train_batches = [self.train_triples[i:i+self.batch_size]
      for i in xrange(0, len(self.train_triples), self.batch_size)]
    valid_batches = [self.valid_triples[i:i+self.batch_size]
      for i in xrange(0, len(self.train_triples), self.batch_size)]
    # Prune batches with invalid number of inputs
    train_batches = filter(lambda b: len(b) == self.batch_size, train_batches)
    valid_batches = filter(lambda b: len(b) == self.batch_size, valid_batches)
    return zip(train_batches, valid_batches)
  
  def size(self):
    return len(self.ix_to_char)


class TrumpTweets(Dataset):
  """Reader for JSON Trump tweets."""
  
  def __init__(self, **kw):
    super(TrumpTweets, self).__init__(**kw)
    tweets = []
    max_chars = self.num_steps + self.gram_order - 1
    for i in xrange(2009, 2018):
      with open('data/trump_tweets/condensed_{}.json'.format(i)) as f:
        file_data = json.load(f)
      for tweet_data in file_data:
        tweet = self.tokenize(tweet_data['text'][:max_chars], add_eos=True)
        # Optimization: instead of concatenating the _PAD token for the shifted
        # tweet, append it and exclude it later (hence the <= rather than <).
        while len(tweet) <= self.num_steps:
          tweet.append(self.char_to_ix['_PAD'])
        tweets.append(tweet)
    
    self.make_triples(tweets)


class WhatsAppChats(Dataset):
  """Reader for WhastApp exported chat history text files."""
  
  def __init__(self, filename='rafi.txt', **kw):
    super(WhatsAppChats, self).__init__(**kw)
    with open('data/whatsapp/' + filename) as f:
      raw_lines = f.readlines()
    
    max_chars = self.num_steps + self.gram_order - 1
    very_ugly_re = r'[0-9]{1,2}/[0-9]{1,2}/[0-9]{1,2}, [0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2} [AP]M: (?:[^:]+): ([^\r]{1,%s})' % max_chars
    
    # Tokenize and chop lines to `num_steps` limit.
    lines = []
    for raw_line in raw_lines:
      match = re.match(very_ugly_re, raw_line)
      if match:
        line = self.tokenize(match.group(1), add_eos=True)
        while len(line) <= self.num_steps:
          line.append(self.char_to_ix['_PAD'])
        lines.append(line)
    
    del raw_lines  # just for memory efficiency
    self.make_triples(lines)


class TextFile(Dataset):
  """Reader for generic text files. Less suitable for seq2seq tasks."""
  
  def __init__(self, filename, **kw):
    super(TextFile, self).__init__(**kw)
    with open('data/' + filename) as f:
      raw_data = f.read()
    data = []
    # Remove the last entry that might have length < `num_steps`.
    max_chars = self.num_steps + self.gram_order - 1
    for i in xrange(0, len(raw_data) - 1, self.num_steps):
      data.append(self.tokenize(raw_data[i:i+max_chars],
        add_eos=True))
    
    del raw_data
    self.make_triples(data)
