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
import re
from textblob import TextBlob
from transformers import pipeline
from dateutil import parser


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class Summarizer:
    def __init__(self):
        self.bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
        self.t5_summarizer = pipeline("summarization", model="t5-base", device=0 if torch.cuda.is_available() else -1)
        self.pegasus_summarizer = pipeline("summarization", model="google/pegasus-xsum", device=0 if torch.cuda.is_available() else -1)

    def bart_summarize(self, text, max_length=25, min_length=12):
        return self.bart_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

    def t5_summarize(self, text, max_length=25, min_length=12):
        return self.t5_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

    def pegasus_summarize(self, text, max_length=25, min_length=12):
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

class CredibilityScorer:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.zero_shot_classifier = pipeline("zero-shot-classification")

    def assess_objectivity(self, text):
        return 1 - TextBlob(text).sentiment.subjectivity

    def evaluate_writing_style(self, text):
        sensationalist_words = ['shocking', 'unbelievable', 'mind-blowing', 'outrageous']
        emotion_words = ['angry', 'sad', 'happy', 'excited', 'scared']

        words = text.split()
        sensationalist_count = sum(word.lower() in text.lower() for word in sensationalist_words)
        emotion_count = sum(word.lower() in text.lower() for word in emotion_words)
        all_caps_ratio = sum(word.isupper() for word in words) / len(words)

        style_score = 1 - (sensationalist_count * 0.1 + emotion_count * 0.05 + all_caps_ratio)
        return max(0, min(style_score, 1))

    def detect_stance(self, text, labels=["factual", "opinion", "rumor"]):
        return self.zero_shot_classifier(text, labels)

    def calculate_user_score(self, karma, account_age_days):
        normalized_karma = min(karma / 10000, 1)
        normalized_age = min(account_age_days / 365, 1)
        return (normalized_karma * 0.5 + normalized_age * 0.5)

    def calculate_credibility_score(self, post, user_score):
        objectivity_score = self.assess_objectivity(post['text'])
        style_score = self.evaluate_writing_style(post['text'])
        sentiment_score = self.sentiment_analyzer(post['text'])[0]['score']
        stance = self.detect_stance(post['text'])
        factual_score = stance['scores'][stance['labels'].index('factual')]

        return (
            objectivity_score * 0.3 +
            style_score * 0.2 +
            sentiment_score * 0.1 +
            factual_score * 0.2 +
            user_score * 0.2
        )

class RedditDataCollector:
    def __init__(self, reddit, subreddits, time_filter='hour', limit=100):
        self.reddit = reddit
        self.subreddits = subreddits
        self.time_filter = time_filter
        self.limit = limit

    async def collect_posts(self, top_n_per_subreddit):
        data = []
        for subreddit_name in self.subreddits:
            subreddit = await self.reddit.subreddit(subreddit_name)
            async for post in subreddit.new(limit=top_n_per_subreddit):
                post_time = datetime.fromtimestamp(post.created_utc)
                if datetime.now() - post_time <= timedelta(hours=0.5):  # Only posts from last 30 minutes
                    data.append({
                        'subreddit': subreddit_name,
                        'title': post.title,
                        'body': post.selftext[:500],
                        'url': post.url,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': post.created_utc,
                        'author': post.author.name
                    })
                    print(f"Collected post from r/{subreddit_name}: {post.title}")
        return pd.DataFrame(data)

def analyze_content_structure(text):
    quotes_count = len(re.findall(r'"([^"]*)"', text))
    numbers_count = len(re.findall(r'\d+', text))
    source_keywords = ['according to', 'sources say', 'reported by']
    sources_count = sum(keyword.lower() in text.lower() for keyword in source_keywords)

    sentences = text.split('.')
    if len(sentences) > 1:
        headline, body = sentences[0], ' '.join(sentences[1:])
        headline_words = set(headline.lower().split())
        body_words = set(body.lower().split())
        coherence = len(headline_words.intersection(body_words)) / len(headline_words)
    else:
        coherence = 1

    structure_score = (quotes_count + numbers_count + sources_count) / 10 + coherence
    return min(structure_score, 1)  # Normalize to 0-1 range

def check_temporal_relevance(text, current_time):
    time_keywords = ['today', 'yesterday', 'last week', 'this month', 'recent']
    time_mentions = sum(keyword.lower() in text.lower() for keyword in time_keywords)

    date_pattern = r'\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2} [A-Za-z]+ \d{2,4}'
    dates = re.findall(date_pattern, text)

    if dates:
        date_diffs = [(current_time - parser.parse(date)).days for date in dates]
        avg_date_diff = sum(date_diffs) / len(date_diffs)
        temporal_score = 1 / (1 + avg_date_diff)
    else:
        temporal_score = time_mentions / len(time_keywords)

    return temporal_score

def calculate_subreddit_score(subscribers):
    return min(subscribers / 1000000, 1)

def classify_post(credibility_score, user_score, subreddit_score, temporal_score, structure_score):
    final_score = (
        credibility_score * 0.5 +
        user_score * 0.1 +
        subreddit_score * 0.1 +
        temporal_score * 0.1 +
        structure_score * 0.2
    )

    if final_score > 0.7:
        return "Likely True"
    elif final_score > 0.4:
        return "Potentially Misleading"
    else:
        return "Needs Further Verification"

async def process_post(post, credibility_scorer, reddit):
    user_info = await get_user_info(reddit, post['author'])
    subreddit_info = await get_subreddit_info(reddit, post['subreddit'])

    user_score = credibility_scorer.calculate_user_score(user_info['karma'], user_info['account_age'])
    credibility_score = credibility_scorer.calculate_credibility_score(post, user_score)
    subreddit_score = calculate_subreddit_score(subreddit_info['subscribers'])
    temporal_score = check_temporal_relevance(post['text'], datetime.now())
    structure_score = analyze_content_structure(post['text'])

    classification = classify_post(credibility_score, user_score, subreddit_score, temporal_score, structure_score)

    return {
        'credibility_score': credibility_score,
        'user_score': user_score,
        'subreddit_score': subreddit_score,
        'temporal_score': temporal_score,
        'structure_score': structure_score,
        'classification': classification
    }

async def get_user_info(reddit, username):
    user = await reddit.redditor(username)
    await user.load()  # Load the user object before accessing attributes
    return {
        'karma': user.link_karma + user.comment_karma,
        'account_age': (datetime.now() - datetime.fromtimestamp(user.created_utc)).days
    }

async def get_subreddit_info(reddit, subreddit_name):
    subreddit = await reddit.subreddit(subreddit_name)
    await subreddit.load()  
    return {
        'subscribers': subreddit.subscribers,
        'active_users': subreddit.active_user_count
    }

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
