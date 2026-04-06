#
# Title: Sutta Web Server (FastAPI + libSQL + Single Page UI)
# Author: @adityapatange_
# Version: 3.0.0
#

import os
import re
from datetime import datetime, timezone
from typing import Literal, Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

try:
    from groq import Groq
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "groq"])
    from groq import Groq

try:
    import libsql
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "libsql"])
    import libsql


# --------------------------------------------------
# ENV
# --------------------------------------------------

load_dotenv(".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LIBSQL_URL = os.getenv("LIBSQL_URL", "file:suttas.db")
LIBSQL_AUTH_TOKEN = os.getenv("LIBSQL_AUTH_TOKEN")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")


# --------------------------------------------------
# APP
# --------------------------------------------------

app = FastAPI(
    title="Sutta Web App",
    description="Generate and store Buddhist Suttas using Groq + libSQL",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# TYPES
# --------------------------------------------------

DifficultyLevel = Literal["beginner", "novice", "advanced", "master"]


class SuttaCreateRequest(BaseModel):
    source: str = Field(..., min_length=3, description="Seed idea for the Sutta")
    difficulty: DifficultyLevel = "beginner"


class SuttaRecord(BaseModel):
    id: int
    title: str
    content: str
    source: str
    difficulty: DifficultyLevel
    created_at: str


class SuttaCreateResponse(BaseModel):
    success: bool
    sutta: SuttaRecord


class SuttaBatchItemInput(BaseModel):
    source: str = Field(..., min_length=3)
    difficulty: DifficultyLevel

    @field_validator("source")
    @classmethod
    def strip_source(cls, v: str) -> str:
        return v.strip()


class SuttaBatchCreateRequest(BaseModel):
    suttas: List[SuttaBatchItemInput] = Field(..., min_length=1, max_length=40)


class SuttaBatchItemResult(BaseModel):
    source: str
    difficulty: DifficultyLevel
    success: bool
    sutta: Optional[SuttaRecord] = None
    error: Optional[str] = None


class SuttaBatchCreateResponse(BaseModel):
    succeeded: int
    failed: int
    results: List[SuttaBatchItemResult]


# --------------------------------------------------
# DB
# --------------------------------------------------


def get_db_connection():
    """
    Supports:
    - local file DB: LIBSQL_URL=file:suttas.db
    - remote Turso/libSQL: LIBSQL_URL=libsql://... or https://...
    """
    if LIBSQL_AUTH_TOKEN:
        return libsql.connect(database=LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    return libsql.connect(database=LIBSQL_URL)


def init_db():
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS suttas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                difficulty TEXT NOT NULL CHECK (difficulty IN ('beginner', 'novice', 'advanced', 'master')),
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
def startup_event():
    init_db()


# --------------------------------------------------
# GROQ
# --------------------------------------------------


def get_groq_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


def build_prompt(source: str, difficulty: DifficultyLevel) -> str:
    difficulty_map = {
        "beginner": "Use simple language, short teachings, gentle examples, and clear moral guidance.",
        "novice": "Use moderate depth, introduce Buddhist concepts gently, and include reflection.",
        "advanced": "Use deeper philosophical insights, Dhamma principles, and contemplative subtlety.",
        "master": "Use profound, highly meditative, non-dual, scripture-like language with strong contemplative force.",
    }

    return f"""
Create a new Buddhist-style Sutta.

SOURCE IDEA:
{source}

DIFFICULTY:
{difficulty} -> {difficulty_map[difficulty]}

STRICT OUTPUT FORMAT:
First line: the Sutta title only
Then a blank line
Then the full Sutta content

RULES:
- No markdown
- No bullets
- No commentary outside the Sutta
- The title must feel scriptural and complete
- The content should read like a coherent spiritual discourse
"""


def parse_sutta_output(raw_text: str) -> tuple[str, str]:
    text = raw_text.strip()

    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    if len(parts) == 2:
        title = parts[0].strip()
        content = parts[1].strip()
    else:
        lines = text.split("\n", 1)
        if len(lines) < 2:
            raise ValueError(
                "Model response did not contain valid title/content format"
            )
        title = lines[0].strip()
        content = lines[1].strip()

    if not title:
        raise ValueError("Generated Sutta title is empty")
    if not content:
        raise ValueError("Generated Sutta content is empty")

    return title, content


def generate_sutta(source: str, difficulty: DifficultyLevel) -> tuple[str, str]:
    client = get_groq_client()
    prompt = build_prompt(source, difficulty)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )

    raw_output = response.choices[0].message.content.strip()
    return parse_sutta_output(raw_output)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------


def row_to_sutta(row) -> SuttaRecord:
    return SuttaRecord(
        id=row[0],
        title=row[1],
        content=row[2],
        source=row[3],
        difficulty=row[4],
        created_at=row[5],
    )


def insert_sutta(
    title: str, content: str, source: str, difficulty: DifficultyLevel
) -> SuttaRecord:
    conn = get_db_connection()
    try:
        created_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO suttas (title, content, source, difficulty, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [title, content, source, difficulty, created_at],
        )
        conn.commit()

        cursor = conn.execute(
            """
            SELECT id, title, content, source, difficulty, created_at
            FROM suttas
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if not row:
            raise RuntimeError("Failed to fetch inserted sutta")

        return row_to_sutta(row)
    finally:
        conn.close()


def fetch_all_suttas(
    difficulty: Optional[str] = None, limit: int = 100
) -> List[SuttaRecord]:
    conn = get_db_connection()
    try:
        if difficulty:
            cursor = conn.execute(
                """
                SELECT id, title, content, source, difficulty, created_at
                FROM suttas
                WHERE difficulty = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                [difficulty, limit],
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, title, content, source, difficulty, created_at
                FROM suttas
                ORDER BY id DESC
                LIMIT ?
                """,
                [limit],
            )

        rows = cursor.fetchall()
        return [row_to_sutta(row) for row in rows]
    finally:
        conn.close()


def fetch_sutta_by_id(sutta_id: int) -> SuttaRecord:
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, title, content, source, difficulty, created_at
            FROM suttas
            WHERE id = ?
            """,
            [sutta_id],
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Sutta not found")
        return row_to_sutta(row)
    finally:
        conn.close()


# --------------------------------------------------
# WEB PAGE ROUTES
# --------------------------------------------------


@app.get("/", include_in_schema=False)
def serve_index():
    if not os.path.exists(INDEX_HTML_PATH):
        raise HTTPException(status_code=500, detail="index.html not found")
    return FileResponse(INDEX_HTML_PATH, media_type="text/html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "sutta-web-app",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "libsql_url": LIBSQL_URL,
    }


# --------------------------------------------------
# API ROUTES
# --------------------------------------------------


@app.post("/api/suttas", response_model=SuttaCreateResponse)
def create_sutta(req: SuttaCreateRequest):
    try:
        title, content = generate_sutta(req.source, req.difficulty)
        sutta = insert_sutta(
            title=title,
            content=content,
            source=req.source,
            difficulty=req.difficulty,
        )
        return {"success": True, "sutta": sutta}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create sutta: {str(e)}")


@app.post("/api/suttas/batch", response_model=SuttaBatchCreateResponse)
def create_suttas_batch(req: SuttaBatchCreateRequest):
    results: List[SuttaBatchItemResult] = []
    succeeded = 0
    failed = 0
    for item in req.suttas:
        try:
            title, content = generate_sutta(item.source, item.difficulty)
            sutta = insert_sutta(
                title=title,
                content=content,
                source=item.source,
                difficulty=item.difficulty,
            )
            results.append(
                SuttaBatchItemResult(
                    source=item.source,
                    difficulty=item.difficulty,
                    success=True,
                    sutta=sutta,
                )
            )
            succeeded += 1
        except Exception as e:
            results.append(
                SuttaBatchItemResult(
                    source=item.source,
                    difficulty=item.difficulty,
                    success=False,
                    error=str(e),
                )
            )
            failed += 1
    return SuttaBatchCreateResponse(
        succeeded=succeeded, failed=failed, results=results
    )


@app.get("/api/suttas", response_model=List[SuttaRecord])
def get_suttas(
    difficulty: Optional[DifficultyLevel] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    try:
        return fetch_all_suttas(difficulty=difficulty, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch suttas: {str(e)}")


@app.get("/api/suttas/{sutta_id}", response_model=SuttaRecord)
def get_sutta(sutta_id: int):
    try:
        return fetch_sutta_by_id(sutta_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch sutta: {str(e)}")
