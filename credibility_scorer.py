from transformers import pipeline
from textblob import TextBlob

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
