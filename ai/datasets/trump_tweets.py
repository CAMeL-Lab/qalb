from six.moves import xrange

import json

from base_datasets import Dataset


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
