# pylint: skip-file

"""Reader for JSON Trump tweet archive."""

import json

from ai.datasets import BaseDataset
from ai.utils import split_train_test


class TrumpTweets(BaseDataset):
  """Reader for JSON Trump tweet archive. Creates character-level LM pairs.
     The original data files can be found here:
     https://github.com/bpb27/trump-tweet-archive/tree/master/data/realdonaldtrump"""
  
  def __init__(self, **kw):
    super().__init__(**kw)
    
    tweets = []
    max_chars = self.num_steps + self.gram_order - 1
    for i in range(2009, 2017 + 1):
      
      with open('data/trump_tweets/condensed_{}.json'.format(i)) as json_file:
        file_data = json.load(json_file)
      
      for tweet_data in file_data:
        tweet = self.tokenize(tweet_data['text'][:max_chars])
        # Optimization: instead of concatenating the _PAD token for the shifted
        # tweet, append it and exclude it later (hence the <= rather than <).
        while len(tweet) <= self.num_steps:
          tweet.append(self.type_to_ix['_PAD'])
        tweets.append(tweet)
    
    del file_data  # just for memory efficiency
    self.make_pairs(tweets)
  
  
  def make_pairs(self, tweets):
    pairs = [(tweet[:-1], tweet[1:]) for tweet in tweets]
    self.train_pairs, self.valid_pairs = split_train_test(pairs)
