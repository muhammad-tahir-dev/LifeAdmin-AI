import os
import streamlit as st

# Streamlit secrets ko os.environ mein daal rahe hain taake backend ko mil sakein
os.environ["SUPABASE_URL"] = st.secrets.get("SUPABASE_URL", "")
os.environ["SUPABASE_KEY"] = st.secrets.get("SUPABASE_KEY", "")
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

st.title("LifeAdmin AI - Working Application")
st.write("Aapka LifeAdmin AI backend successfully connect ho chuka hai!")

# Backend import kar rahe hain
import sys
sys.path.append(os.path.abspath("lifeadmin-backend"))
from main import app

st.success("Application is live and running!")
