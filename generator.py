from google import genai
from config import GEMINI_API_KEY

# genai.configure(api_key=GEMINI_API_KEY)

class Generator:
    def __init__(self):
        # self.model = genai.GenerativeModel("gemini-2.5-pro")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def generate(self, query, context_chunks):
        context = "\n".join(context_chunks)
        prompt = f"""Answer the question using only the context below.
Context:
{context}

Question: {query}
Answer:"""
        # response = self.model.generate_content(prompt)
        # return response.text
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text