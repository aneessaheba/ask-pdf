import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
print(os.getenv("GROQ_API_KEY"))