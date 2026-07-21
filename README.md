# 📰 Fake News Classifier

A machine learning system that classifies news articles as **Real** or **Fake**, trained on the WELFake dataset (72K+ articles) using TF-IDF + Logistic Regression. Deployed as a live FastAPI backend and Streamlit web app.

**🔗 Live Demo:** [fakenewsclassisier-sd8n2chzld7s3hpcjj78cz.streamlit.app](https://fakenewsclassisier-sd8n2chzld7s3hpcjj78cz.streamlit.app/)  
**🔗 API Docs:** [fake-news-classisier.onrender.com/docs](https://fake-news-classisier.onrender.com/docs)

---

## 📊 Overview

| | |
|---|---|
| **Task** | Binary text classification — Real vs. Fake news |
| **Dataset** | [WELFake](https://zenodo.org/records/4561253) — 72,134 articles (35,028 real, 37,106 fake) |
| **Model** | TF-IDF (50,000 features) → Logistic Regression |
| **Accuracy** | 95.3% |
| **Backend** | FastAPI, deployed on Render |
| **Frontend** | Streamlit, deployed on Streamlit Community Cloud |

---

## 🏗️ Architecture

```
┌────────────────────┐        HTTP POST         ┌────────────────────┐
│   Streamlit App     │ ─────────────────────────▶ │   FastAPI Backend    │
│ (Streamlit Cloud)    │ ◀───────────────────────── │      (Render)          │
└────────────────────┘        JSON response       └────────────────────┘
                                                              │
                                                              ▼
                                                  fake_news_classifier.joblib
                                                  (sklearn Pipeline: TF-IDF +
                                                   Logistic Regression)
```

The frontend and backend are deployed independently and communicate over HTTP — the same pattern as my [Titanic Survival Predictor](https://github.com/AmeyDande/Titanic-analysis.git) project.

---

## 📂 Project Structure

```
Fake_News_Classisier/
├── app.py                       # FastAPI backend (prediction API)
├── streamlit_app.py             # Streamlit frontend (UI)
├── requirements.txt             # All dependencies (backend + frontend)
├── fake_news_classifier.joblib  # Trained Pipeline (TF-IDF + Logistic Regression)
├── notebooks/
│   ├── EDA.ipynb                # Exploratory data analysis
│   └── train.py                 # Training script
└── README.md
```

---

## ⚙️ How It Works

**1. Preprocessing** — `clean_text()` in `app.py` mirrors the exact steps used during training:
- Strip HTML tags
- Strip URLs
- Remove non-alphabetic characters
- Lowercase everything
- Remove English stopwords (NLTK)

**2. Input** — Headline (`title`) and article body (`text`) are combined into a single string before cleaning, matching how the model was trained (`title + " " + text`).

**3. Prediction** — The cleaned text is passed directly into the saved `Pipeline`, which handles TF-IDF vectorization and classification in one step.

**4. Output** — A label (`Real`/`Fake`), the predicted class, and confidence probabilities for both classes.

---

## 📥 Dataset

This project uses the **WELFake** dataset (72,134 labeled news articles, merged from four existing fake-news datasets: Kaggle, McIntire, Reuters, and BuzzFeed Political).

**Download from either:**
- Kaggle: [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- Zenodo (original source): [zenodo.org/records/4561253](https://zenodo.org/records/4561253)

After downloading, place `WELFake_Dataset.csv` in the same directory as `notebooks/train.py` before running the training script. The dataset is not included in this repo due to its size — only the trained model artifact (`fake_news_classifier.joblib`) is committed.

---

## 🚀 Running Locally

```bash
git clone https://github.com/AmeyDande/Fake_News_Classisier.git
cd Fake_News_Classisier
pip install -r requirements.txt
```

**Start the backend:**
```bash
uvicorn app:app --reload
```
API docs available at `http://localhost:8000/docs`

**Start the frontend** (in a separate terminal):
```bash
streamlit run streamlit_app.py
```

> Note: `streamlit_app.py` points to the live Render URL by default. Change `API_URL` to `http://localhost:8000` if you want the local frontend to hit your local backend instead.

---

## ☁️ Deployment

### Backend → Render
| Setting | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Runtime | Python 3 |

### Frontend → Streamlit Community Cloud
| Setting | Value |
|---|---|
| Main file | `streamlit_app.py` |
| Repo | Same repo as backend |

---

## 📡 API Reference

**POST** `/predict`

Request:
```json
{
  "title": "Senate Passes Infrastructure Bill in Bipartisan Vote",
  "text": "The Senate voted 69-30 on Tuesday to pass a sweeping infrastructure package..."
}
```

Response:
```json
{
  "prediction": "Real",
  "label": 1,
  "confidence": 0.94,
  "probabilities": { "Fake": 0.06, "Real": 0.94 }
}
```

**GET** `/health` — uptime check, returns `{"status": "healthy"}`

---

## 🧠 Model Details

- **Pipeline:** `TfidfVectorizer(max_features=50000, stop_words='english')` → `LogisticRegression(max_iter=1000)`
- **Labels:** `0 = Fake`, `1 = Real` (WELFake convention)
- **Evaluation:** 95.3% accuracy on a held-out test split (80/20 train-test)

### ⚠️ Known Limitation

This model classifies based on **lexical and stylistic patterns learned from its specific training sources**, not independent fact-checking. WELFake's labels come from which original dataset an article was sourced from (Kaggle, McIntire, Reuters, BuzzFeed Political), so the model can pick up on formatting and vocabulary quirks of those sources rather than the actual truthfulness of a claim.

In practice, this means:
- Well-written misinformation that mimics the tone of "real" training sources can be misclassified as Real
- Legitimate news that doesn't closely match the vocabulary patterns in the "real" training data can be misclassified as Fake

This is a common limitation of TF-IDF + linear model approaches to fake news detection, and a key motivation for the next stage of this project — fine-tuning a transformer model (BERT) that captures semantic meaning rather than just word frequency.

---

## 🗺️ Roadmap

This project is the first step in a larger NLP/LLM portfolio arc:

**Fake News Classifier** (this project) → BERT Fine-Tuning → Semantic Search Engine → RAG Chatbot

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `NLTK` · `FastAPI` · `Streamlit` · `Render` · `Streamlit Community Cloud`
