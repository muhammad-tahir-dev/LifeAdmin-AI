import os
import streamlit as st
import streamlit.components.v1 as components

# Streamlit secrets ko os.environ mein daal rahe hain taake backend ko mil sakein
os.environ["SUPABASE_URL"] = st.secrets.get("SUPABASE_URL", "")
os.environ["SUPABASE_KEY"] = st.secrets.get("SUPABASE_KEY", "")
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

# HTML frontend file ko read karke Streamlit par load kar rahe hain
frontend_path = os.path.join("frontend", "lifeadmin-frontend.html")
if os.path.exists(frontend_path):
    with open(frontend_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=850, scrolling=True)
else:
    st.error("Frontend HTML file nahi mili! Directory check karein.")

# Backend FastAPI ko import kar rahe hain
import sys
sys.path.append(os.path.abspath("lifeadmin-backend"))
from main import app
