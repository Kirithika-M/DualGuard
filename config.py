from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TOP_K = 5
MAX_CORRECTION_ATTEMPTS = 3
FAITHFULNESS_THRESHOLD = 0.75