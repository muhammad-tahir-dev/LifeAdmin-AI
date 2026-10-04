import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables explicitly
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

groq_api_key = os.getenv("GROQ_API_KEY")

def generate_ai_plan(goal_title: str, days: int, budget: int):
    try:
        # Groq ka is waqt ka active aur stable model
        llm = ChatGroq(
            temperature=0.3,
            model_name="openai/gpt-oss-120b",
            groq_api_key=groq_api_key
        )
        
        prompt = f"""
        Create a detailed, phase-wise daily action plan for this goal:
        - Goal: {goal_title}
        - Timeframe: {days} days
        - Budget: {budget} PKR
        
        Provide a structured, practical, and motivating plan with milestones.
        """
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating AI plan: {str(e)}"