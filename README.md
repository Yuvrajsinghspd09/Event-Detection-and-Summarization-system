# Event-Detection-and-Summarization-system

# 🌐 Reddit News Summarizer

Harness the power of AI to summarize the latest news from Reddit! This tool collects top posts from popular news subreddits and generates concise summaries using state-of-the-art language models.

## 🚀 Features

- Collects recent posts from major news subreddits
- Utilizes multiple summarization techniques:
  - BART (Facebook)
  - T5 (Google)
  - PEGASUS (Google)
  - Custom extractive summarization
- Asynchronous data collection for improved performance
- Customizable number of top news items to summarize

## 🛠️ Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/reddit-news-summarizer.git
   cd reddit-news-summarizer
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Reddit API credentials:
   - Create a Reddit account and navigate to https://www.reddit.com/prefs/apps
   - Create a new app, select "script" as the type
   - Note down your client ID and client secret

4. Update the `client_id` and `client_secret` in the `main()` function with your credentials.

## 💻 Usage

Run the script:

```
python reddit_news_summarizer.py
```

When prompted, enter the number of top news items you want to summarize.

## 🧠 How It Works

1. **Data Collection**: Fetches recent posts from specified news subreddits using PRAW (Python Reddit API Wrapper).
2. **Summarization**:
   - BART: Facebook's bidirectional transformer model
   - T5: Google's Text-to-Text Transfer Transformer
   - PEGASUS: Google's abstractive summarization model
   - Extractive: Custom algorithm based on word frequency
3. **Output**: Displays summaries, post details, and relevant metadata for each top news item.

## 🔧 Customization

- Modify the `subreddits` list in `main()` to target different subreddits.
- Adjust the `time_filter` and `limit` parameters in `RedditDataCollector` to change the post collection criteria.
- Fine-tune summarization parameters in the `Summarizer` class methods.

## 📊 Performance

The script utilizes asyncio for efficient data collection and leverages GPU acceleration (if available) for summarization tasks.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [PRAW](https://praw.readthedocs.io/) for Reddit API interaction
- [Hugging Face Transformers](https://huggingface.co/transformers/) for pre-trained summarization models
- [NLTK](https://www.nltk.org/) for natural language processing tasks

Happy summarizing! 📰✨
