from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from database import supabase
from agents import generate_ai_plan
from part5 import router as member5_router

app = FastAPI(title="LifeAdmin AI Backend", version="2.0.0")

# Member 5: PDF/CV analysis, finance, verification and adaptive planning.
# This router is additive and does not modify existing Member 2/3/4 routes or database schema.
app.include_router(member5_router, prefix="/part5", tags=["Member 5"])

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GoalCreate(BaseModel):
    title: str
    days: int = 30
    budget: Optional[int] = 0

class TaskUpdate(BaseModel):
    done: Optional[bool] = None
    skipped: Optional[bool] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to LifeAdmin AI Backend (Supabase + CrewAI)!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "status_code": 200}

# --- GOALS ENDPOINTS ---

@app.get("/goals/")
def get_goals():
    try:
        response = supabase.table("goals").select("*").execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/goals/")
def create_goal(goal: GoalCreate):
    try:
        # 1. Goal ko Supabase database mein save karein
        goal_response = supabase.table("goals").insert({
            "title": goal.title,
            "days": goal.days,
            "budget": goal.budget
        }).execute()
        
        goal_data = goal_response.data[0] if goal_response.data else {}

        # 2. CrewAI + Groq agents ke zariye intelligent plan generate karein
        ai_result = generate_ai_plan(
            goal_title=goal.title,
            days=goal.days,
            budget=goal.budget
        )
        
        return {
            "status": "success",
            "message": "Goal successfully saved and AI Plan generated via CrewAI!",
            "goal": goal_data,
            "ai_plan": str(ai_result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/goals/{goal_id}")
def delete_goal(goal_id: str):
    try:
        supabase.table("tasks").delete().eq("goal_id", goal_id).execute()
        response = supabase.table("goals").delete().eq("id", goal_id).execute()
        return {
            "status": "success",
            "message": f"Goal {goal_id} and its tasks successfully deleted!",
            "data": response.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- TASKS ENDPOINTS ---

@app.get("/goals/{goal_id}/tasks")
def get_tasks(goal_id: str):
    try:
        response = supabase.table("tasks").select("*").eq("goal_id", goal_id).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/tasks/{task_id}")
def update_task_status(task_id: str, task_update: TaskUpdate):
    try:
        update_data = {}
        if task_update.done is not None:
            update_data["done"] = task_update.done
        if task_update.skipped is not None:
            update_data["skipped"] = task_update.skipped
            
        response = supabase.table("tasks").update(update_data).eq("id", task_id).execute()
        return {
            "status": "success",
            "message": "Task status successfully updated!",
            "data": response.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))