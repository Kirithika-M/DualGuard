from retriever import Retriever
from generator import Generator
from critic import Critic
from reformulator import Reformulator
from config import TOP_K, MAX_CORRECTION_ATTEMPTS

class DualGuardPipeline:
    def __init__(self, documents):
        self.retriever = Retriever(documents)
        self.generator = Generator()
        self.critic = Critic()
        self.reformulator = Reformulator()

    def run(self, query):
        chunks = self.retriever.retrieve(query, TOP_K)
        draft_answer = self.generator.generate(query, chunks)
        sentences = self.critic.split_sentences(draft_answer)

        final_output = []

        for sentence in sentences:
            result = self.critic.classify(sentence, chunks, query)
            attempts = 0
            current_sentence = sentence
            current_chunks = chunks
            current_query = query

            while result["label"] == "HIGH_HALLUCINATION" and attempts < MAX_CORRECTION_ATTEMPTS:
                new_query = self.reformulator.reformulate(current_query, current_sentence)
                new_chunks = self.retriever.retrieve(new_query, TOP_K)
                regenerated = self.generator.generate(new_query, new_chunks)

                current_sentence = self.critic.split_sentences(regenerated)[0]
                result = self.critic.classify(current_sentence, new_chunks, query)  # always check against ORIGINAL query for relevance
                current_chunks = new_chunks
                current_query = new_query
                attempts += 1

            if result["label"] == "LOW_HALLUCINATION":
                cited_chunk = current_chunks[result["best_chunk_idx"]]
                final_output.append({
                    "sentence": current_sentence,
                    "trust": "HIGH_TRUST",
                    "faith_score": round(result["faith_score"], 3),
                    "rel_score": round(result["rel_score"], 3),
                    "citation": cited_chunk
                })
            else:
                final_output.append({
                    "sentence": current_sentence,
                    "trust": "LOW_TRUST_FLAGGED",
                    "faith_score": round(result["faith_score"], 3),
                    "rel_score": round(result["rel_score"], 3),
                    "citation": None
                })

        return final_output