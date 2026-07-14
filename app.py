import re
import joblib
import nltk
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
from pydantic import BaseModel, Field


app = FastAPI(
    title="Fake News Classifier API",
    description="TF-IDF + Logistic Regression pipeline trained on WELFake dataset (95.3% accuracy)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Streamlit app's exact URL once deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PIPELINE_PATH = "fake_news_classifier.joblib"

try:
    pipeline = joblib.load(PIPELINE_PATH)
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model artifact not found: {e}. Make sure '{PIPELINE_PATH}' is in "
        f"the same directory as main.py."
    )


try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# WELFake convention: 0 = fake, 1 = real
LABEL_MAP = {0: "Fake", 1: "Real"}



def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)



class NewsInput(BaseModel):
    title: str = Field(
        default="",
        description="News headline (optional but improves accuracy — the model was trained on title + text combined)",
    )
    text: str = Field(
        ...,
        min_length=10,
        description="The news article body to classify",
        json_schema_extra={"example": "Scientists discover new exoplanet in habitable zone"},
    )


class PredictionOutput(BaseModel):
    prediction: str
    label: int
    confidence: float
    probabilities: dict



@app.get("/")
def root():
    return {
        "message": "Fake News Classifier API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict_news(input_data: NewsInput):
    try:
        combined = f"{input_data.title} {input_data.text}".strip()
        if not combined:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")

        cleaned = clean_text(combined)

        pred_label = int(pipeline.predict([cleaned])[0])
        pred_proba = pipeline.predict_proba([cleaned])[0]

        probabilities = {
            LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(pred_proba)
        }
        confidence = round(float(np.max(pred_proba)), 4)

        return PredictionOutput(
            prediction=LABEL_MAP[pred_label],
            label=pred_label,
            confidence=confidence,
            probabilities=probabilities,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)