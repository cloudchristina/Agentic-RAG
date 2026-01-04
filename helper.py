import os
from dotenv import load_dotenv, find_dotenv

def load_env():
    """Load environment variables from a .env file."""
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        print("No .env file found.")

def get_groq_api_key():
    """Retrieve the Groq API key from environment variables."""
    load_env()
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return api_key