#
# Title: Sutta Server (FastAPI - Full Version)
# Author: @adityapatange_
# Version: 2.0.0
#

import os
import re
from datetime import datetime
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
try:
    from groq import Groq
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "groq"])
    from groq import Groq

# -----------------------------
# ENV SETUP
# -----------------------------

load_dotenv(dotenv_path=".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise Exception("❌ GROQ_API_KEY not found in .env")

# -----------------------------
# FASTAPI INIT
# -----------------------------

app = FastAPI(
    title="Sutta API 🧘",
    description="Generate and manage Buddhist Suttas",
    version="2.0.0",
)

# -----------------------------
# CONSTANTS
# -----------------------------

DifficultyLevel = Literal["beginner", "novice", "advanced", "master"]

SUTTA_BASE_DIR = "suttas"

# Ensure directories exist
for level in ["beginner", "novice", "advanced", "master"]:
    os.makedirs(os.path.join(SUTTA_BASE_DIR, level), exist_ok=True)

# -----------------------------
# MODELS
# -----------------------------


class SuttaRequest(BaseModel):
    source: str = Field(..., description="Seed idea for the Sutta")
    difficulty: DifficultyLevel = "beginner"


class BatchSuttaRequest(BaseModel):
    suttas: List[SuttaRequest]


class SuttaResponse(BaseModel):
    sutta_name: str
    sutta_content: str
    difficulty: DifficultyLevel
    file_path: str


# -----------------------------
# GROQ CLIENT
# -----------------------------


def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)


# -----------------------------
# UTILITIES
# -----------------------------


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def build_prompt(source: str, difficulty: DifficultyLevel) -> str:
    difficulty_map = {
        "beginner": "Use simple language, relatable metaphors, and short teachings.",
        "novice": "Use moderate depth, introduce Buddhist concepts gently.",
        "advanced": "Use deeper philosophical insights, reference Dhamma principles.",
        "master": "Use profound, non-dual, abstract and deeply meditative insights like ancient Suttas.",
    }

    return f"""
Create a new Buddhist Sutta.

Source:
{source}

Difficulty:
{difficulty} → {difficulty_map[difficulty]}

STRICT FORMAT:
<sutta_name>
<sutta_content>

RULES:
- No extra commentary
- Name must be on first line
- Content follows after newline
- Style: Authentic Buddhist scripture
"""


def parse_sutta_output(text: str):
    lines = text.strip().split("\n", 1)

    if len(lines) < 2:
        raise Exception("Invalid Sutta format from model")

    return lines[0].strip(), lines[1].strip()


def save_sutta(name: str, content: str, difficulty: str) -> str:
    slug = slugify(name)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    filename = f"{slug}_{timestamp}.txt"
    path = os.path.join(SUTTA_BASE_DIR, difficulty, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{name}\n\n{content}")

    return path


# -----------------------------
# CORE GENERATION
# -----------------------------


def generate_sutta(source: str, difficulty: DifficultyLevel):
    client = get_groq_client()

    prompt = build_prompt(source, difficulty)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )

    raw_output = response.choices[0].message.content.strip()

    name, content = parse_sutta_output(raw_output)

    file_path = save_sutta(name, content, difficulty)

    return {
        "sutta_name": name,
        "sutta_content": content,
        "file_path": file_path,
    }


# -----------------------------
# ROUTES
# -----------------------------


@app.get("/")
def health():
    return {"status": "🧘 Sutta API running", "timestamp": datetime.utcnow()}


@app.post("/sutta", response_model=SuttaResponse)
def create_sutta(req: SuttaRequest):
    try:
        result = generate_sutta(req.source, req.difficulty)

        return {
            "sutta_name": result["sutta_name"],
            "sutta_content": result["sutta_content"],
            "difficulty": req.difficulty,
            "file_path": result["file_path"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/suttas/batch")
def create_batch_suttas(req: BatchSuttaRequest):
    results = []

    for item in req.suttas:
        try:
            result = generate_sutta(item.source, item.difficulty)

            results.append(
                {
                    "success": True,
                    "data": {
                        "sutta_name": result["sutta_name"],
                        "difficulty": item.difficulty,
                        "file_path": result["file_path"],
                    },
                }
            )

        except Exception as e:
            results.append({"success": False, "error": str(e), "input": item.dict()})

    return {"count": len(results), "results": results}


@app.get("/suttas/{difficulty}")
def list_suttas(difficulty: DifficultyLevel):
    dir_path = os.path.join(SUTTA_BASE_DIR, difficulty)

    files = os.listdir(dir_path)

    return {"difficulty": difficulty, "count": len(files), "files": files}


@app.get("/sutta/read/{difficulty}/{filename}")
def read_sutta(difficulty: DifficultyLevel, filename: str):
    path = os.path.join(SUTTA_BASE_DIR, difficulty, filename)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sutta not found")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return {"filename": filename, "difficulty": difficulty, "content": content}
