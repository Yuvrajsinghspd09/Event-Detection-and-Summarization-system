# 2. Data Collection
class TwitterDataCollector:
    def __init__(self, api, max_tweets, output_file):
        self.api = api
        self.max_tweets = max_tweets
        self.output_file = output_file

    def collect_sample_stream(self):
        class SampleStreamListener(tweepy.StreamingClient):
            def __init__(self, bearer_token, max_tweets, output_file):
                super().__init__(bearer_token)
                self.max_tweets = max_tweets
                self.tweet_count = 0
                self.output_file = output_file

            def on_tweet(self, tweet):
                if self.tweet_count < self.max_tweets:
                    with open(self.output_file, 'a') as f:
                        json.dump(tweet.data, f)
                        f.write('\n')
                    self.tweet_count += 1
                    return True
                else:
                    self.disconnect()
                    return False

            def on_error(self, status):
                print(f"Error: {status}")
                return True

        # You need to use a bearer token for StreamingClient
        bearer_token = "YOUR_BEARER_TOKEN"
        listener = SampleStreamListener(bearer_token, self.max_tweets, self.output_file)
        listener.sample(tweet_fields=['author_id', 'created_at', 'lang'])
