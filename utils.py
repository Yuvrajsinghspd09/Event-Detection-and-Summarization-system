import re
from datetime import datetime
from dateutil import parser

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
    await subreddit.load()  # Load the subreddit object before accessing attributes
    return {
        'subscribers': subreddit.subscribers,
        'active_users': subreddit.active_user_count
    }
