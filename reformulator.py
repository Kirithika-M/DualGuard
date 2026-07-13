import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

class Reformulator:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def reformulate(self, original_query, flagged_sentence):
        prompt = f"""The following sentence was flagged as unsupported by evidence:
"{flagged_sentence}"

Original question: {original_query}

Rewrite the original question into a more specific search query
that would help retrieve better supporting evidence for this claim.
Return only the rewritten query."""
        response = self.model.generate_content(prompt)
        return response.text.strip()