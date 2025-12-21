from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# -----------------------------
# Internal modules (merged logic)
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from collections import defaultdict

nltk.download("vader_lexicon")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="AI Trust Insights Backend")

# -----------------------------
# CORS (FIXES YOUR ERROR)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# NLP Feature Extraction
# -----------------------------
def extract_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()

    df["sentiment"] = df["confidence_note"].fillna("").apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    df["skepticism_flag"] = df["confidence_note"].str.contains(
        "not|wrong|missed|override|human|manual|uncertain|mismatch",
        case=False,
        na=False,
    ).astype(int)

    return df

# -----------------------------
# ML Model
# -----------------------------
def train_trust_model(df: pd.DataFrame):
    X = df[["sentiment", "skepticism_flag"]]
    y = df["override"]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )

    pipeline.fit(X, y)
    return pipeline


def explain_model(pipeline):
    model = pipeline.named_steps["model"]
    return {
        "sentiment_weight": float(model.coef_[0][0]),
        "skepticism_weight": float(model.coef_[0][1]),
    }

# -----------------------------
# Trust Metrics
# -----------------------------
def calculate_trust_metrics(df: pd.DataFrame):
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W").astype(str)

    metrics = (
        df.groupby("week")
        .agg(
            override_rate=("override", "mean"),
            trust_score=("override", lambda x: 1 - x.mean()),
            total_cases=("override", "count"),
        )
        .reset_index()
    )

    return metrics

# -----------------------------
# RAG + FAISS
# -----------------------------
class TrustRAG:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.texts = []

    def build_index(self, texts):
        self.texts = texts
        embeddings = self.model.encode(texts)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def classify_theme(self, text: str):
        t = text.lower()

        if any(x in t for x in ["wrong", "incorrect", "inaccurate"]):
            return "Perceived model inaccuracy"
        if any(x in t for x in ["context", "mismatch", "role"]):
            return "Context mismatch"
        if any(x in t for x in ["human", "manual", "experience"]):
            return "Preference for human judgment"
        return "General skepticism"

    def build_explanations(self, notes):
        theme_map = defaultdict(list)

        for note in notes:
            theme = self.classify_theme(note)
            theme_map[theme].append(note)

        explanations = []
        for theme, items in theme_map.items():
            explanations.append(
                {
                    "theme": theme,
                    "count": len(items),
                    "example": items[0],
                }
            )

        explanations.sort(key=lambda x: x["count"], reverse=True)
        return explanations

# -----------------------------
# Executive Summary
# -----------------------------
def generate_executive_summary(metrics, ml_weights, rag_explanations):
    low_trust = metrics[metrics["trust_score"] < 0.5]

    summary = []

    if len(low_trust) > 0:
        summary.append(
            "Trust declined during periods with increased override rates."
        )
    else:
        summary.append(
            "Trust levels remained stable throughout the observed period."
        )

    if rag_explanations:
        summary.append(
            f"RAG analysis highlights '{rag_explanations[0]['theme']}' as the dominant driver of trust decay."
        )

    strongest_signal = max(
        ml_weights, key=lambda k: abs(ml_weights[k])
    )
    summary.append(
        f"ML analysis indicates '{strongest_signal}' as the strongest predictor of trust failure."
    )

    if len(low_trust) > 0:
        summary.append(
            "Targeted intervention is recommended during low-trust periods to restore confidence."
        )

    return " ".join(summary)

# -----------------------------
# MAIN ENDPOINT
# -----------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Override flag
    df["override"] = df["model_decision"] != df["human_decision"]

    # NLP
    df = extract_nlp_features(df)

    # ML
    pipeline = train_trust_model(df)
    ml_weights = explain_model(pipeline)
    df["override_risk"] = pipeline.predict_proba(
        df[["sentiment", "skepticism_flag"]]
    )[:, 1]

    # Top risky cases
    top_risks = (
        df.sort_values("override_risk", ascending=False)
        .head(5)[["confidence_note", "override_risk"]]
        .to_dict(orient="records")
    )

    # Metrics
    metrics = calculate_trust_metrics(df)

    # RAG
    rag = TrustRAG()
    notes = df["confidence_note"].dropna().tolist()
    rag.build_index(notes)
    rag_explanations = rag.build_explanations(notes)

    # Executive summary
    executive_summary = generate_executive_summary(
        metrics, ml_weights, rag_explanations
    )

    return {
        "metrics": metrics.to_dict(orient="records"),
        "ml_weights": ml_weights,
        "top_risks": top_risks,
        "rag_explanations": rag_explanations,
        "executive_summary": executive_summary,
    }
