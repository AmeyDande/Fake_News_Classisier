# 📰 Fake News Classifier

A machine learning system that classifies news articles as **Real** or **Fake**, trained on the WELFake dataset (72K+ articles). Two model versions exist: a **TF-IDF + Logistic Regression** pipeline (v1, currently deployed) and a fine-tuned **DistilBERT** transformer (v2, in progress). v1 is deployed as a live FastAPI backend and Streamlit web app.

**🔗 Live Demo:** [fakenewsclassisier-6menjrgvjvohz969yc8kxp.streamlit.app](https://fakenewsclassisier-sd8n2chzld7s3hpcjj78cz.streamlit.app/)  
**🔗 API Docs:** [fake-news-classisier.onrender.com/docs](https://fake-news-classisier.onrender.com/docs)  
**🔗 v2 Model (Hugging Face Hub):** [GhostFaith/distilbert-fakenews](https://huggingface.co/GhostFaith/distilbert-fakenews)

---

## 📊 Overview

| | |
|---|---|
| **Task** | Binary text classification — Real vs. Fake news |
| **Dataset** | [WELFake](https://zenodo.org/records/4561253) — 72,134 articles (35,028 real, 37,106 fake) |
| **Model (v1, deployed)** | TF-IDF (50,000 features) → Logistic Regression — 95.3% accuracy |
| **Model (v2, in progress)** | Fine-tuned `distilbert-base-uncased` — 99.1% accuracy, 0.99 F1 |
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
├── README.md
├── requirements.txt
├── app.py                          # FastAPI backend
├── streamlit_app.py                # Streamlit frontend
├── models/
│   └── fake_news_classifier.joblib # v1 artifact, isolated from code
└── notebooks/
    ├── EDA.ipynb
    ├── train_tfidf.py              # renamed from using-tfidf-v1.py
    └── finetune_distilbert.ipynb   # renamed from using-distilbest-v2.ipynb 
```

> The v2 DistilBERT weights aren't committed to this repo — they're hosted on the Hugging Face Hub ([GhostFaith/distilbert-fakenews](https://huggingface.co/GhostFaith/distilbert-fakenews)) and loaded from there. v2 isn't wired into `app.py` yet; the deployed API still serves v1.v2 isn't deployed yet because Render's free-tier web service caps out at 512MB RAM, which isn't enough to load PyTorch, transformers, and the DistilBERT weights alongside the existing FastAPI app — deployment is blocked on either quantizing the model or moving inference to a host with more memory (e.g. Hugging Face Spaces).


---

## ⚙️ How It Works (v1 — deployed)

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

## 🔬 How the DistilBERT Fine-Tuning Works (v2)

**Why DistilBERT.** TF-IDF + Logistic Regression scores words independently, so it can't tell "the bill passed easily" from "the bill barely passed" — it has no sense of word order or context. DistilBERT is a distilled version of BERT (40% fewer parameters, ~60% faster) that keeps most of BERT's language understanding, making it a practical choice for fine-tuning on a single Colab GPU without the full compute cost of BERT-base.

**1. No manual text cleaning.** Unlike v1's `clean_text()` pipeline, the raw `title + text` string is passed directly to the tokenizer — no stopword removal, no lowercasing, no stripping punctuation. Transformer tokenizers are trained to make use of casing, punctuation, and stopwords as context signals, so stripping them out (as TF-IDF requires) would throw away information the model can otherwise use.

**2. Tokenization.** Each example is tokenized with `truncation=True, padding="max_length", max_length=512` — articles are cut off at 512 tokens (DistilBERT's context limit) and padded to a uniform length so they batch together.

**3. Data split.** Unlike v1's 80/20 train-test split, v2 uses an **80/10/10 train/validation/test** split. The extra validation set exists specifically to pick the best checkpoint during training (see below) rather than just training blind for a fixed number of steps.

**4. Fine-tuning setup** — trained via the Hugging Face `Trainer` API with a few deliberate choices:
- **2 epochs** — transformers fine-tune fast and tend to overfit quickly on a dataset this size; 2 passes was enough to reach strong validation performance without memorizing the training set.
- **`fp16=True`** — mixed-precision training, roughly halves memory use and speeds up training on GPU with negligible accuracy cost.
- **`per_device_train_batch_size=8`** — kept small deliberately, since `max_length=512` padding on every example is memory-heavy; a larger batch size would have exceeded the Colab GPU's memory at this sequence length.
- **`warmup_steps=500`** — the learning rate ramps up gradually for the first 500 steps instead of starting at full strength, which helps stabilize early training when the classification head is still randomly initialized.
- **`weight_decay=0.01`** — light L2 regularization to reduce overfitting risk given the small epoch count.
- **`load_best_model_at_end=True`, `metric_for_best_model='f1'`** — the checkpoint from whichever epoch scored highest validation F1 is kept, not necessarily the final epoch. F1 was chosen over raw accuracy so precision and recall on both classes are weighted, not just overall correctness.

**5. Evaluation and shipping.** After training, the best checkpoint is evaluated once on the held-out test split (99.1% accuracy, 0.99 F1 — see [Known Limitation](#️-known-limitation-applies-to-both-versions) for how to read that number), then pushed to the Hugging Face Hub (`model.push_to_hub(...)`) rather than committed to this repo, since transformer weights are far larger than the TF-IDF pipeline's joblib file.

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

### v1 — TF-IDF + Logistic Regression (deployed)

- **Pipeline:** `TfidfVectorizer(max_features=50000, stop_words='english')` → `LogisticRegression(max_iter=1000)`
- **Labels:** `0 = Fake`, `1 = Real` (WELFake convention)
- **Evaluation:** 95.3% accuracy on a held-out test split (80/20 train-test)

### v2 — Fine-tuned DistilBERT (in progress)

- **Base model:** `distilbert-base-uncased`, fine-tuned end-to-end (2 epochs, fp16, batch size 8/16)
- **Labels:** `0 = Fake`, `1 = Real` (same convention as v1)
- **Evaluation:** 99.1% accuracy, 0.99 F1 on a held-out test split (80/10/10 train/val/test)
- **Weights:** hosted on the Hugging Face Hub — [GhostFaith/distilbert-fakenews](https://huggingface.co/GhostFaith/distilbert-fakenews)
- **Status:** trained and evaluated, not yet integrated into the deployed FastAPI backend

### ⚠️ Known Limitation (applies to both versions)

Both models classify based on **patterns learned from WELFake's specific training sources**, not independent fact-checking. WELFake's labels come from which original dataset an article was sourced from (Kaggle, McIntire, Reuters, BuzzFeed Political), so a model can pick up on formatting and vocabulary quirks of those sources rather than the actual truthfulness of a claim.

In practice, this means:
- Well-written misinformation that mimics the tone of "real" training sources can be misclassified as Real
- Legitimate news that doesn't closely match the vocabulary patterns in the "real" training data can be misclassified as Fake

This is a known limitation of TF-IDF + linear model approaches, and was the original motivation for fine-tuning a transformer (v2) that captures semantic meaning rather than just word frequency. v2's much higher accuracy (99.1% vs. 95.3%) is a promising signal, but hasn't yet been validated on out-of-distribution articles (i.e. news from sources outside WELFake) — so it's not yet confirmed whether the gain reflects genuine semantic understanding or just a stronger fit to the same source-level artifacts. Out-of-distribution testing is a planned next step before v2 replaces v1 in production.


---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `NLTK` · `PyTorch` · `Transformers` · `Hugging Face Hub` · `FastAPI` · `Streamlit` · `Render` · `Streamlit Community Cloud`
