# 2. Data Collection
class TwitterDataCollector:
    def __init__(self, api, max_tweets, output_file):
        self.api = api
        self.max_tweets = max_tweets
        self.output_file = output_file

    def collect_sample_stream(self):
        class SampleStreamListener(tweepy.Stream):
            def __init__(self, api_key, api_secret_key, access_token, access_token_secret, max_tweets, output_file):
                super().__init__(api_key, api_secret_key, access_token, access_token_secret)
                self.max_tweets = max_tweets
                self.tweet_count = 0
                self.output_file = output_file

            def on_data(self, data):
                if self.tweet_count < self.max_tweets:
                    with open(self.output_file, 'a') as f:
                        f.write(data)
                    self.tweet_count += 1
                    return True
                else:
                    return False

            def on_error(self, status):
                print(f"Error: {status}")
                return True

        listener = SampleStreamListener(
            self.api.auth.consumer_key,
            self.api.auth.consumer_secret,
            self.api.auth.access_token,
            self.api.auth.access_token_secret,
            self.max_tweets,
            self.output_file
        )
        listener.filter(languages=['en'])
