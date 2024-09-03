import asyncio
import pandas as pd
from datetime import datetime
import time
from summarizer import Summarizer
from reddit_collector import RedditDataCollector
import asyncpraw
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def main(top_n=5):
    start_time = time.time()
    print("Starting data collection...")

    reddit = asyncpraw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
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
