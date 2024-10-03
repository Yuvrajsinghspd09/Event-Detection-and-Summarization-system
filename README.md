# NewsGuardian: 
## AI-Powered Reddit News Detector, Summarizer, and Credibility Analyzer🌐🔍🛡️

Hey there! Welcome to NewsGuardian, your smart companion for navigating the wild world of Reddit news. This tool doesn't just find news - it breaks it down, summarizes it, and helps you figure out what's trustworthy.

## What's This All About? 🤔

Ever felt lost in the sea of Reddit news, unsure what to believe? NewsGuardian's got your back! Here's what it does:

- Hunts down fresh, hot news from top Reddit communities
- Creates quick, smart summaries using cutting-edge AI
- Gives each story a "trust score" to help you spot the real deal
- Breaks down the writing style and how the content is structured
- Checks if the news is recent and relevant
- Looks at how reliable the poster and the community are
- Labels posts as "Likely True", "Potentially Misleading", or "Needs a Fact-Check"


## Getting Started 🚀

1. Clone this repo to your machine
2. Install the stuff we need:
pip install -r requirements.txt


3. Set up your Reddit API creds (check out Reddit's dev site for this)
4. Update the `client_id` and `client_secret` in `main.py` with your info

## To Run This Thing! 💻

Just fire up the main script:
python main.py

It'll ask how many top news items you want to see from each subreddit. Type in a number and let it generate!

## What's Under the Hood? 🔧

- `main.py`: The boss - runs the whole show
- `reddit_collector.py`: Our Reddit post hunter
- `summarizer.py`: The brains behind our summaries
- `credibility_scorer.py`: Figures out how trustworthy a post is
- `utils.py`: A bunch of helpful tools we use along the way

## Wanna Tweak It? 🛠️

Feel free to mess around with the code! You can:
- Change which subreddits we look at in `main.py`
- Adjust how we pick posts in `reddit_collector.py`
- Fiddle with the summary settings in `summarizer.py`
- Tweak how we score credibility in `credibility_scorer.py`
