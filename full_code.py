import asyncio
import torch
import asyncpraw
import pandas as pd
from transformers import pipeline
import nest_asyncio
import time
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class Summarizer:
    def __init__(self):
        self.bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
        self.t5_summarizer = pipeline("summarization", model="t5-base", device=0 if torch.cuda.is_available() else -1)
        self.pegasus_summarizer = pipeline("summarization", model="google/pegasus-xsum", device=0 if torch.cuda.is_available() else -1)

    def bart_summarize(self, text, max_length=150, min_length=50):
        return self.bart_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

    def t5_summarize(self, text, max_length=150, min_length=50):
        return self.t5_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

    def pegasus_summarize(self, text, max_length=150, min_length=50):
        return self.pegasus_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

    def extractive_summarize(self, text, num_sentences=3):
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        freq_dist = FreqDist(words)
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in freq_dist:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = freq_dist[word]
                    else:
                        sentence_scores[sentence] += freq_dist[word]
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        summary = ' '.join(summary_sentences)
        return summary

class RedditDataCollector:
    def __init__(self, reddit, subreddits, time_filter='hour', limit=100):
        self.reddit = reddit
        self.subreddits = subreddits
        self.time_filter = time_filter
        self.limit = limit

    async def collect_posts(self):
        data = []
        for subreddit_name in self.subreddits:
            subreddit = await self.reddit.subreddit(subreddit_name)
            async for post in subreddit.new(limit=self.limit):
                post_time = datetime.fromtimestamp(post.created_utc)
                if datetime.now() - post_time <= timedelta(hours=0.5):  # Only posts from last 30 minutes
                    data.append({
                        'subreddit': subreddit_name,
                        'title': post.title,
                        'body': post.selftext[:500],
                        'url': post.url,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': post.created_utc
                    })
                    print(f"Collected post from r/{subreddit_name}: {post.title}")
        return pd.DataFrame(data)

async def main(top_n=5):
    start_time = time.time()
    print("Starting data collection...")

    reddit = asyncpraw.Reddit(
        client_id="MXvTTPJnwZCjHRXarkpQKA",
        client_secret="uaFWby-c9cCGl24F8MnAEp9fa5izvQ",
        user_agent="event-summarizer"
    )

    subreddits = ["news", "worldnews", "politics", "technology", "science"]
    collector = RedditDataCollector(reddit, subreddits, time_filter='hour', limit=100)
    df = await collector.collect_posts()
    df['text'] = df['title'] + ' ' + df['body']

    print(f"Data collection completed in {time.time() - start_time:.2f} seconds")
    print("Initializing summarizer...")

    summarizer = Summarizer()

    print(f"Processing top {top_n} posts...")
    top_posts = df.nlargest(top_n, 'score')
    
    for _, post in top_posts.iterrows():
        print(f"Subreddit: r/{post['subreddit']}")
        print(f"Title: {post['title']}")
        print(f"BART Summary: {summarizer.bart_summarize(post['text'])}")
        print(f"T5 Summary: {summarizer.t5_summarize(post['text'])}")
        print(f"PEGASUS Summary: {summarizer.pegasus_summarize(post['text'])}")
        print(f"Extractive Summary: {summarizer.extractive_summarize(post['text'])}")
        print(f"URL: {post['url']}")
        print(f"Score: {post['score']}")
        print(f"Number of comments: {post['num_comments']}")
        print(f"Created: {datetime.fromtimestamp(post['created_utc']).strftime('%Y-%m-%d %H:%M:%S')}\n")

    await reddit.close()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

def run_main(top_n=5):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(main(top_n))

if __name__ == "__main__":
    top_n = int(input("Enter the number of top news items you want to see: "))
    run_main(top_n)










'''
Imports and Initialization:
The code starts by importing necessary libraries and initializing NLTK data. It uses various libraries for asynchronous programming, natural language processing, and machine learning.
Summarizer Class:
This class contains methods for different summarization techniques using pre-trained models (BART, T5, PEGASUS) and an extractive summarization method.
RedditDataCollector Class:
This class is responsible for collecting data from Reddit using the PRAW library.
Main Function:
The main function orchestrates the entire process of collecting data and generating summaries.

'''
