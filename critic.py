from sentence_transformers import SentenceTransformer, util
import nltk
import joblib
import numpy as np

nltk.download("punkt", quiet=True)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

class Critic:
    def __init__(self, model_path="hallucination_classifier.pkl"):
        self.classifier = joblib.load(model_path)

    def split_sentences(self, text):
        return nltk.sent_tokenize(text)

    def faithfulness_score(self, sentence, context_chunks):
        sent_emb = embedder.encode(sentence, convert_to_tensor=True)
        chunk_embs = embedder.encode(context_chunks, convert_to_tensor=True)
        scores = util.cos_sim(sent_emb, chunk_embs)
        best_idx = int(scores.argmax())
        return float(scores.max()), best_idx

    def relevance_score(self, sentence, query):
        sent_emb = embedder.encode(sentence, convert_to_tensor=True)
        query_emb = embedder.encode(query, convert_to_tensor=True)
        return float(util.cos_sim(sent_emb, query_emb))

    def classify(self, sentence, context_chunks, query):
        faith_score, best_chunk_idx = self.faithfulness_score(sentence, context_chunks)
        rel_score = self.relevance_score(sentence, query)

        features = np.array([[faith_score, rel_score]])
        pred = self.classifier.predict(features)[0]  # 0 = LOW, 1 = HIGH
        label = "HIGH_HALLUCINATION" if pred == 1 else "LOW_HALLUCINATION"

        return {
            "label": label,
            "faith_score": faith_score,
            "rel_score": rel_score,
            "best_chunk_idx": best_chunk_idx
        }