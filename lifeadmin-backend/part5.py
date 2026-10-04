from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Optional

import fitz  # PyMuPDF
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from research.llm import LLMUnavailable, groq_call, parse_json_object

router = APIRouter()

# ----------------------------- Models ---------------------------------

class FinanceTask(BaseModel):
    title: str
    cost: float = Field(default=0, ge=0)
    optional: bool = False
    done: bool = False
    skipped: bool = False


class FinanceRequest(BaseModel):
    budget: float = Field(default=0, ge=0)
    tasks: list[FinanceTask] = Field(default_factory=list)


class VerificationItem(BaseModel):
    field: str
    value: Any = None
    required: bool = True


class VerificationRequest(BaseModel):
    items: list[VerificationItem] = Field(default_factory=list)


class AdaptiveTask(BaseModel):
    id: str
    title: str
    due: int = Field(default=1, ge=1)
    off: float = Field(default=0, ge=0)
    cost: float = Field(default=0, ge=0)
    optional: bool = False
    done: bool = False
    skipped: bool = False


class AdaptivePlanRequest(BaseModel):
    days: int = Field(..., ge=1)
    budget: float = Field(default=0, ge=0)
    previous_days: int = Field(..., ge=1)
    previous_budget: float = Field(default=0, ge=0)
    tasks: list[AdaptiveTask] = Field(default_factory=list)


# ----------------------------- Helpers --------------------------------

MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_TEXT_CHARS_FOR_AI = 24000


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(file_bytes: bytes) -> dict[str, Any]:
    """Extract text and basic metadata from a PDF without changing any DB data."""
    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")
    if len(file_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF is too large. Maximum supported size is 15 MB.")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("The uploaded file is not a valid PDF.") from exc

    try:
        pages = []
        for index, page in enumerate(doc):
            text = _clean_text(page.get_text("text"))
            if text:
                pages.append({"page": index + 1, "text": text})
        metadata = doc.metadata or {}
        page_count = len(doc)
    finally:
        doc.close()

    full_text = "\n\n".join(p["text"] for p in pages)
    return {
        "page_count": page_count,
        "text": full_text,
        "pages_with_text": len(pages),
        "metadata": {
            "title": metadata.get("title") or "",
            "author": metadata.get("author") or "",
            "subject": metadata.get("subject") or "",
        },
    }


def _document_prompt(filename: str, extracted_text: str) -> str:
    return f"""Analyze this uploaded document for LifeAdmin AI.

File name: {filename}

DOCUMENT TEXT:
{extracted_text[:MAX_TEXT_CHARS_FOR_AI]}

Return ONLY a JSON object with these keys:
{{
  "document_type": "CV|resume|identity|financial|education|other",
  "summary": "short summary",
  "person": {{"name": "", "email": "", "phone": "", "location": ""}},
  "education": [],
  "experience": [],
  "skills": [],
  "certifications": [],
  "financial_information": [],
  "important_dates": [],
  "missing_or_unclear_information": [],
  "verification_required": []
}}

Rules:
- Extract only information actually present in the document.
- Do not invent values.
- Use empty strings/lists when information is absent.
- Keep financial_information generic; never infer bank passwords, PINs, or secret credentials.
- verification_required should list important extracted claims that should be checked before they are used in a plan.
"""


def analyze_with_ai(filename: str, text: str) -> dict[str, Any]:
    try:
        raw = groq_call(
            _document_prompt(filename, text),
            json_mode=True,
            temperature=0,
            max_tokens=1800,
            retries=1,
        )
        return parse_json_object(raw)
    except (LLMUnavailable, ValueError, TypeError) as exc:
        return {
            "document_type": "other",
            "summary": "PDF text was extracted successfully, but AI analysis is currently unavailable.",
            "person": {},
            "education": [],
            "experience": [],
            "skills": [],
            "certifications": [],
            "financial_information": [],
            "important_dates": [],
            "missing_or_unclear_information": ["AI analysis unavailable"],
            "verification_required": [],
            "ai_status": "unavailable",
            "ai_error": str(exc)[:200],
        }


def _money(value: float) -> float:
    return round(float(value or 0), 2)


# ----------------------------- Document --------------------------------

@router.post("/document/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(default=None),
):
    """Extract PDF text with PyMuPDF and optionally structure it with Groq.

    This endpoint does not write to Supabase. The caller can decide whether the
    extracted result should be persisted or sent to Member 4's RAG ingestion.
    """
    filename = file.filename or "document.pdf"
    content_type = (file.content_type or "").lower()
    if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Part 5 currently accepts PDF files only.")

    data = await file.read()
    try:
        extracted = extract_pdf(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not extracted["text"]:
        return {
            "status": "PARTIAL",
            "message": "PDF opened successfully, but no selectable text was found. A scanned/image-only PDF needs OCR.",
            "filename": filename,
            "user_id": user_id,
            "page_count": extracted["page_count"],
            "pages_with_text": 0,
            "extracted_text": "",
            "analysis": None,
        }

    analysis = analyze_with_ai(filename, extracted["text"])
    return {
        "status": "SUCCESS" if analysis.get("ai_status") != "unavailable" else "PARTIAL",
        "filename": filename,
        "user_id": user_id,
        "page_count": extracted["page_count"],
        "pages_with_text": extracted["pages_with_text"],
        "metadata": extracted["metadata"],
        "extracted_text": extracted["text"],
        "analysis": analysis,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


# ----------------------------- Finance ---------------------------------

@router.post("/finance/analyze")
def analyze_finance(request: FinanceRequest):
    active = [t for t in request.tasks if not t.skipped]
    required = [t for t in active if not t.optional]
    optional = [t for t in active if t.optional]
    planned = _money(sum(t.cost for t in active))
    required_cost = _money(sum(t.cost for t in required))
    optional_cost = _money(sum(t.cost for t in optional))
    spent = _money(sum(t.cost for t in active if t.done))
    remaining = _money(request.budget - planned) if request.budget else 0
    over_budget = bool(request.budget and planned > request.budget)

    recommendations = []
    if over_budget:
        for task in sorted((t for t in optional if not t.done), key=lambda x: x.cost, reverse=True):
            recommendations.append({
                "action": "skip_optional",
                "task": task.title,
                "saving": _money(task.cost),
            })
            planned -= task.cost
            if planned <= request.budget:
                break
        if planned > request.budget:
            recommendations.append({
                "action": "review_required_costs",
                "message": f"Required costs alone are PKR {required_cost:,.2f}.",
            })
    else:
        recommendations.append({"action": "keep_plan", "message": "Current planned cost fits the budget."})

    return {
        "status": "SUCCESS",
        "budget": _money(request.budget),
        "planned_cost": _money(sum(t.cost for t in active)),
        "required_cost": required_cost,
        "optional_cost": optional_cost,
        "spent": spent,
        "remaining": remaining,
        "over_budget": over_budget,
        "recommendations": recommendations,
    }


# ----------------------------- Verification ----------------------------

@router.post("/verification/check")
def verification_check(request: VerificationRequest):
    results = []
    for item in request.items:
        value = item.value
        field = item.field.strip() or "unnamed field"
        present = value is not None and str(value).strip() != ""
        reason = "Present" if present else "Missing"
        status = "PASS" if present else ("REVIEW" if item.required else "OPTIONAL")

        if field.lower() == "email" and present:
            valid = bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", str(value).strip()))
            status = "PASS" if valid else "REVIEW"
            reason = "Valid email format" if valid else "Email format should be checked"
        elif field.lower() in {"phone", "mobile"} and present:
            digits = re.sub(r"\D", "", str(value))
            status = "PASS" if 7 <= len(digits) <= 15 else "REVIEW"
            reason = "Plausible phone number format" if status == "PASS" else "Phone format should be checked"

        results.append({"field": field, "status": status, "reason": reason, "value": value})

    review_count = sum(r["status"] == "REVIEW" for r in results)
    missing_required = sum(r["status"] == "REVIEW" and not str(r["value"] or "").strip() for r in results)
    return {
        "status": "SUCCESS",
        "overall": "REVIEW" if review_count else "PASS",
        "review_count": review_count,
        "missing_required": missing_required,
        "items": results,
        "note": "Format/completeness checks are not proof that a fact is true. Important claims should be confirmed from an official source.",
    }


# ----------------------------- Adaptive plan ----------------------------

def adapt_plan(request: AdaptivePlanRequest) -> dict[str, Any]:
    tasks = deepcopy([t.model_dump() for t in request.tasks])
    changes: list[str] = []

    if request.days != request.previous_days:
        for task in tasks:
            task["due"] = max(1, round(float(task.get("off", 0)) * request.days))
        changes.append(
            f"Deadline changed from {request.previous_days} to {request.days} days; task due days were rescaled."
        )
        if request.days < request.previous_days * 0.5:
            changes.append("The new deadline is much shorter; optional tasks may need to be skipped.")

    # Clear previous automatic skips only; never touch a user's manual skip.
    for task in tasks:
        if task.get("skipped") and task.get("auto_skipped"):
            task["skipped"] = False
            task["auto_skipped"] = False

    if request.budget != request.previous_budget:
        changes.append(
            f"Budget changed from PKR {request.previous_budget:,.2f} to PKR {request.budget:,.2f}."
        )

    if request.budget > 0:
        def active_cost() -> float:
            return sum(float(t.get("cost", 0) or 0) for t in tasks if not t.get("skipped"))

        for task in sorted(
            [t for t in tasks if t.get("optional") and not t.get("done") and float(t.get("cost", 0) or 0) > 0],
            key=lambda x: float(x.get("cost", 0) or 0),
            reverse=True,
        ):
            if active_cost() <= request.budget:
                break
            task["skipped"] = True
            task["auto_skipped"] = True
            changes.append(f'Skipped "{task.get("title", "Untitled task")}" to fit the budget.')

        if active_cost() > request.budget:
            changes.append(
                f"Required costs are PKR {active_cost():,.2f}, still above the budget. Review required costs."
            )

    planned = _money(sum(float(t.get("cost", 0) or 0) for t in tasks if not t.get("skipped")))
    return {
        "status": "SUCCESS",
        "days": request.days,
        "budget": _money(request.budget),
        "planned_cost": planned,
        "tasks": tasks,
        "changes": changes,
    }


@router.post("/adaptive-plan")
def adaptive_plan(request: AdaptivePlanRequest):
    return adapt_plan(request)
