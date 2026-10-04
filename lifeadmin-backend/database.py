import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Explicitly find and load the .env file from the project root
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase URL or Key in environment variables!")

# Initialize Supabase client securely
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)