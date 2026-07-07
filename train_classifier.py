import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_features(answer, knowledge, question):
    """Same two signals your Critic already computes at inference time."""
    ans_emb = embedder.encode(answer, convert_to_tensor=True)
    know_emb = embedder.encode(knowledge, convert_to_tensor=True)
    ques_emb = embedder.encode(question, convert_to_tensor=True)

    faith_score = float(util.cos_sim(ans_emb, know_emb))
    rel_score = float(util.cos_sim(ans_emb, ques_emb))
    return [faith_score, rel_score]

def build_dataset(path="data/data/qa_data.json", limit=None):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read().strip()

    # HaluEval's qa_data.json is one JSON object per line (JSONL), not a single array
    records = [json.loads(line) for line in raw.split("\n") if line.strip()]
    if limit:
        records = records[:limit]

    X, y = [], []
    for r in records:
        knowledge = r["knowledge"]
        question = r["question"]
        right_answer = r["right_answer"]
        hallucinated_answer = r["hallucinated_answer"]

        # Label 0 = LOW_HALLUCINATION (faithful), 1 = HIGH_HALLUCINATION
        X.append(extract_features(right_answer, knowledge, question))
        y.append(0)

        X.append(extract_features(hallucinated_answer, knowledge, question))
        y.append(1)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Building feature dataset from HaluEval qa_data.json...")
    # Start with a subset (e.g. 2000) while testing -- full 10K pairs = 20K rows takes a while on CPU
    X, y = build_dataset(limit=2000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["LOW_HALLUCINATION", "HIGH_HALLUCINATION"]))

    joblib.dump(model, "hallucination_classifier.pkl")
    print("Saved trained classifier to hallucination_classifier.pkl")