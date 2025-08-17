import sqlite3
import time
from typing import Dict, List, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    status TEXT,
    prompt TEXT,
    model_name TEXT,
    base_response TEXT,
    reasoning_response TEXT,
    base_score INTEGER,
    reasoning_score INTEGER,
    base_feedback TEXT,
    reasoning_feedback TEXT,
    choice TEXT,
    last_updated REAL,
    error_log TEXT
)
"""

def get_connection(db_config: Dict) -> sqlite3.Connection:
    """Return a sqlite3 connection based on config."""
    if db_config.get("type", "sqlite") != "sqlite":
        raise NotImplementedError("Only sqlite databases are supported in this reference implementation.")
    path = db_config.get("path", "tasks.db")
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(conn: sqlite3.Connection) -> None:
    """Initialize the tasks table if it does not exist."""
    conn.execute(SCHEMA)
    conn.commit()

def insert_task(conn: sqlite3.Connection, task_id: str, prompt: str, model_name: str) -> None:
    """Insert a new task if it does not already exist."""
    now = time.time()
    conn.execute(
        """
        INSERT OR IGNORE INTO tasks (task_id, status, prompt, model_name, last_updated)
        VALUES (?, 'PENDING_GENERATION', ?, ?, ?)
        """,
        (task_id, prompt, model_name, now),
    )
    conn.commit()

def fetch_and_lock_tasks(conn: sqlite3.Connection, from_status: str, to_status: str, limit: int) -> List[sqlite3.Row]:
    """Fetch tasks with a given status and immediately lock them by updating their status.

    This prevents multiple workers from processing the same task concurrently.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM tasks WHERE status=? LIMIT ?", (from_status, limit)
    )
    rows = cur.fetchall()
    if not rows:
        return []
    ids = [r["task_id"] for r in rows]
    placeholders = ",".join(["?"] * len(ids))
    cur.execute(
        f"UPDATE tasks SET status=?, last_updated=? WHERE task_id IN ({placeholders})",
        [to_status, time.time(), *ids],
    )
    conn.commit()
    return rows

def update_generation_success(conn: sqlite3.Connection, task_id: str, base_resp: str, reasoning_resp: str) -> None:
    conn.execute(
        """
        UPDATE tasks SET status='GENERATION_COMPLETE', base_response=?, reasoning_response=?, last_updated=?
        WHERE task_id=?
        """,
        (base_resp, reasoning_resp, time.time(), task_id),
    )
    conn.commit()

def update_generation_failure(conn: sqlite3.Connection, task_id: str, error: str) -> None:
    conn.execute(
        """
        UPDATE tasks SET status='FAILED_GENERATION', error_log=?, last_updated=?
        WHERE task_id=?
        """,
        (error, time.time(), task_id),
    )
    conn.commit()

def update_evaluation_success(
    conn: sqlite3.Connection,
    task_id: str,
    base_score: int,
    reasoning_score: int,
    base_feedback: str,
    reasoning_feedback: str,
    choice: str,
) -> None:
    conn.execute(
        """
        UPDATE tasks SET status='COMPLETE', base_score=?, reasoning_score=?,
            base_feedback=?, reasoning_feedback=?, choice=?, last_updated=?
        WHERE task_id=?
        """,
        (
            base_score,
            reasoning_score,
            base_feedback,
            reasoning_feedback,
            choice,
            time.time(),
            task_id,
        ),
    )
    conn.commit()

def update_evaluation_failure(conn: sqlite3.Connection, task_id: str, error: str) -> None:
    conn.execute(
        """
        UPDATE tasks SET status='FAILED_EVALUATION', error_log=?, last_updated=?
        WHERE task_id=?
        """,
        (error, time.time(), task_id),
    )
    conn.commit()
