from sentence_transformers import SentenceTransformer, util
import nltk
from config import FAITHFULNESS_THRESHOLD, RELEVANCE_THRESHOLD

nltk.download("punkt", quiet=True)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

class Critic:
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

        is_faithful = faith_score >= FAITHFULNESS_THRESHOLD
        is_relevant = rel_score >= RELEVANCE_THRESHOLD

        if is_faithful and is_relevant:
            label = "LOW_HALLUCINATION"
        else:
            label = "HIGH_HALLUCINATION"

        return {
            "label": label,
            "faith_score": faith_score,
            "rel_score": rel_score,
            "best_chunk_idx": best_chunk_idx
        }