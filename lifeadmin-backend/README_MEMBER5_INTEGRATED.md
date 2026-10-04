# LifeAdmin AI — Member 5 Integrated Version

This folder is the original LifeAdmin backend (Members 2–4) with Member 5 added.

## What was changed
- Added `part5.py` for:
  - PDF/CV extraction with PyMuPDF
  - Groq-based document structuring using the existing Member 4 Groq wrapper
  - Finance analysis
  - Verification/completeness checks
  - Adaptive deadline/budget planning
- Added Member 5 router to `main.py` under `/part5`.
- Added `requirements.txt` for the complete backend.
- Added `test_part5.py`.
- Existing database.py, agents.py, research package, and existing API routes were retained.
- Member 5 does NOT create, delete, or alter Supabase tables/data.

## Important: keep your existing .env
Do not replace your working `.env`. The ZIP intentionally contains `.env.example` only.
Your existing `.env` should remain in this folder with your working Supabase/Groq/Tavily keys.

## Install
Open CMD/PowerShell in this folder:

```bash
python -m pip install -r requirements.txt
```

If you already have the previous environment activated, just run the command above. It will add missing packages and keep existing packages.

## Run
```bash
uvicorn main:app --reload
```

Then open:
`http://127.0.0.1:8000/docs`

## Member 5 endpoints
- `POST /part5/document/analyze` — upload a PDF/CV and analyze it.
- `POST /part5/finance/analyze` — calculate planned/required/optional/spent cost and budget recommendations.
- `POST /part5/verification/check` — completeness and format checks.
- `POST /part5/adaptive-plan` — rescale deadlines and automatically skip optional tasks when needed to fit budget.

## Test without changing Supabase data
```bash
python test_part5.py
```
The test creates a PDF in memory and mocks the Groq response. It does not write to Supabase.

## Existing data safety
Member 5 is additive. It does not alter the existing `goals`, `tasks`, or Member 4 research code. Document analysis is returned to the caller; it is not automatically inserted into Supabase.
