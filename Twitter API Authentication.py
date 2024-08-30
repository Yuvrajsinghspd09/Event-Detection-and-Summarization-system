import tweepy


class TwitterAuth:
  def __init__(self, api_key, api_secret_key, access_token , access_token_secret):
    self.api_key=api_key
    self.api_secret_key=api_secret_key
    self.access_token = access_token
    self.access_token_secret= access_token_secret

   #tweepy is imported at start to deal with authentication in this function
  def authenticate(self):
    auth = tweepy.OAuthHandler(self.api_key, self.api_secret_key)
    auth.set_access_token(self.access_token, self.access_token_secret)
    api = tweepy.API(auth)
    return api


# Usage
auth = TwitterAuth("your_api_key", "your_api_secret_key", "your_access_token", "your_access_token_secret")
api = auth.authenticate()
