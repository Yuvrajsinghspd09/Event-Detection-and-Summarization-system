class TwitterDataCollector:
  def __init__(self, api, max_tweets, output_file):
    self.api = api
    self.max_tweets = max_tweets
    self.output_file = output_file
    
    def collect_sample_stream(self):
      class SampleStreamListener(tweepy.StreamListener):
        def __init__(self, max_tweets, output_file):
          self.max_tweets = max_tweets
          self.tweet_count = 0
          self.output_file = output_file

        def on_data(self, data):
          if self.tweet_count<self.max_tweets :
            with open(self.output_file, 'a') as f:
              f.write(data)
            self.tweet_count+=1
            return True
          else:
            return False

        def on_error(self,status):
          print(f"Error: {status}")

      listener = SampleStreamListener()
      stream = tweepy.Stream(auth = self.api.auth, listener = listener)
      stream.sample(languages=['en'])

