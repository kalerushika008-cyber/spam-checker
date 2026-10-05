# 📩 Spam Message Checker

A beginner-friendly machine learning web app that tells you whether a message is **spam or not spam**.

Built with **Python, Streamlit, pandas and numpy**. The Naive Bayes model is written from scratch with numpy.

## Features
- Check a message and see spam probability and risk level
- Red/green word highlighting
- Teach the AI with your own examples
- Upload a CSV dataset
- Bulk check and download results
- History and stats

## How to run

```bash
git clone https://github.com/YOUR-USERNAME/spam-checker.git
cd spam-checker
python -m venv venv
source venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements.txt
streamlit run app.py
```

## How it works
The app counts words in spam and non-spam messages and uses Naive Bayes
probability to guess which group a new message belongs to.

## Tech stack
Python · Streamlit · pandas · numpy# spam-checker
