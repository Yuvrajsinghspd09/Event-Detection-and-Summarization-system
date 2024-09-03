import torch
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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
