import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import spacy

class TweetProcessor:
  def __init__(self):
    self.stop_words = set(stopwords.words('english'))

  def clean_text(self,text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text.strip()

  def tokenize(self, text):
      return word_tokenize(text)

  def remove_stop_words(self, tokens):
      return [token for token in tokens if token not in self.stop_words]

  def named_entity_recognition(self, text):
      doc = nlp(text)
      return [(ent.text, ent.label_) for ent in doc.ents]

  def preprocess(self,text):
    cleaned_text = self.clean_text(text)
    tokens= self.word_tokenize(cleaned_text)
    tokens_without_stopwords = self.remove_stop_words(cleaned_text)
    entities = self.named_entity_recognition(cleaned_text)
    return { 'cleaned_text' : cleaned_text,
              'tokens' : tokens_without_stopwords,
              'entities' : entities,
           }





if __name__ == "__main__":
  auth = TwitterAuth(
        api_key="your_api_key",
        api_secret_key="your_api_secret_key",
        access_token="your_access_token",
        access_token_secret="your_access_token_secret"
  )
  api = auth.authenticate()

  collector = TwitterDataCollector(api, max_tweets=10000, output_file="tweets.json")
  collector.collect_sample_stream()
  preprocessor = TweetPreprocessor()

    # Example of preprocessing a single tweet
    with open("tweets.json", "r") as f:
        tweet = json.loads(f.readline())
    
    preprocessed_tweet = preprocessor.preprocess(tweet['text'])
    print("Preprocessed tweet:", preprocessed_tweet)
   
 

  
