import asyncio
import time
from datetime import datetime
import pandas as pd
import asyncpraw
from reddit_collector import RedditDataCollector
from summarizer import Summarizer
from credibility_scorer import CredibilityScorer
from utils import process_post, get_user_info, get_subreddit_info

async def main():
    start_time = time.time()
    print("Starting data collection...")

    reddit = asyncpraw.Reddit(
        client_id="",
        client_secret="",
        user_agent="event-summarizer"
    )

    subreddits = ["news", "worldnews", "politics", "technology", "science"]
    top_n_per_subreddit = int(input("Enter the number of top news items you want to see from each subreddit: "))
    collector = RedditDataCollector(reddit, subreddits, time_filter='hour', limit=top_n_per_subreddit)
    df = await collector.collect_posts(top_n_per_subreddit)
    df['text'] = df['title'] + ' ' + df['body']

    print(f"Data collection completed in {time.time() - start_time:.2f} seconds")
    print("Initializing summarizer and credibility scorer...")

    summarizer = Summarizer()
    credibility_scorer = CredibilityScorer()

    for subreddit_name in subreddits:
        print(f"Posts from r/{subreddit_name}:")
        top_posts = df[df['subreddit'] == subreddit_name].nlargest(top_n_per_subreddit, 'score')

        for _, post in top_posts.iterrows():
            print(f"Title: {post['title']}")
            print(f"BART Summary: {summarizer.bart_summarize(post['text'])}")
            print(f"T5 Summary: {summarizer.t5_summarize(post['text'])}")
            print(f"PEGASUS Summary: {summarizer.pegasus_summarize(post['text'])}")
            print(f"Extractive Summary: {summarizer.extractive_summarize(post['text'])}")

            result = await process_post(post, credibility_scorer, reddit)
            print(f"Credibility Score: {result['credibility_score']:.2f}")
            print(f"User Score: {result['user_score']:.2f}")
            print(f"Subreddit Score: {result['subreddit_score']:.2f}")
            print(f"Temporal Score: {result['temporal_score']:.2f}")
            print(f"Structure Score: {result['structure_score']:.2f}")
            print(f"Classification: {result['classification']}")

            print(f"URL: {post['url']}")
            print(f"Score: {post['score']}")
            print(f"Number of comments: {post['num_comments']}")
            print(f"Created: {datetime.fromtimestamp(post['created_utc']).strftime('%Y-%m-%d %H:%M:%S')}\n")

    await reddit.close()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

def run_main():
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(main())

if __name__ == "__main__":
    run_main()
