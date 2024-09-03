import pandas as pd
from datetime import datetime, timedelta

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
