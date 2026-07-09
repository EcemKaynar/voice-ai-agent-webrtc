import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

METRICS_DB_PATH = os.getenv("METRICS_DB_PATH", "data/voice_metrics.db")


def get_connection():
    db_path = Path(METRICS_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row

    return connection


def init_metrics_db():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS voice_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            peer_id TEXT,
            transcript TEXT,
            answer TEXT,
            audio_input_path TEXT,
            audio_output_path TEXT,
            stt_success INTEGER,
            llm_success INTEGER,
            tts_success INTEGER,
            stt_latency_ms INTEGER,
            llm_first_token_ms INTEGER,
            llm_total_ms INTEGER,
            tts_first_byte_ms INTEGER,
            tts_total_ms INTEGER,
            total_pipeline_ms INTEGER,
            llm_model TEXT,
            tts_voice TEXT,
            errors_json TEXT
        )
        """
    )

    connection.commit()
    connection.close()


def save_voice_metric(
    peer_id,
    transcript,
    answer,
    audio_input_path=None,
    audio_output_path=None,
    stt_success=None,
    llm_success=None,
    tts_success=None,
    stt_latency_ms=None,
    llm_first_token_ms=None,
    llm_total_ms=None,
    tts_first_byte_ms=None,
    tts_total_ms=None,
    total_pipeline_ms=None,
    llm_model=None,
    tts_voice=None,
    errors=None
):
    init_metrics_db()

    errors = errors or {}

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO voice_interactions (
            created_at,
            peer_id,
            transcript,
            answer,
            audio_input_path,
            audio_output_path,
            stt_success,
            llm_success,
            tts_success,
            stt_latency_ms,
            llm_first_token_ms,
            llm_total_ms,
            tts_first_byte_ms,
            tts_total_ms,
            total_pipeline_ms,
            llm_model,
            tts_voice,
            errors_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            peer_id,
            transcript,
            answer,
            str(audio_input_path) if audio_input_path else None,
            str(audio_output_path) if audio_output_path else None,
            1 if stt_success else 0,
            1 if llm_success else 0,
            1 if tts_success else 0,
            stt_latency_ms,
            llm_first_token_ms,
            llm_total_ms,
            tts_first_byte_ms,
            tts_total_ms,
            total_pipeline_ms,
            llm_model,
            tts_voice,
            json.dumps(errors, ensure_ascii=False)
        )
    )

    metric_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return metric_id


def get_recent_voice_metrics(limit=10):
    init_metrics_db()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            created_at,
            peer_id,
            transcript,
            answer,
            audio_input_path,
            audio_output_path,
            stt_success,
            llm_success,
            tts_success,
            stt_latency_ms,
            llm_first_token_ms,
            llm_total_ms,
            tts_first_byte_ms,
            tts_total_ms,
            total_pipeline_ms,
            llm_model,
            tts_voice,
            errors_json
        FROM voice_interactions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    connection.close()

    items = []

    for row in rows:
        item = dict(row)

        try:
            item["errors"] = json.loads(item.get("errors_json") or "{}")
        except Exception:
            item["errors"] = {}

        item.pop("errors_json", None)

        items.append(item)

    return items