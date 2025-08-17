"""Generation worker for Reasoning Model Evaluation Pipeline.

The worker polls the task database for tasks with status
``PENDING_GENERATION``.  For each task it generates two responses using a
placeholder generation function (standing in for vLLM) and stores the
results back to the database.

This worker is intentionally simple: it processes tasks sequentially and
terminates once no pending tasks remain.  It can be run in multiple
instances to simulate replicas.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import yaml

from task_db import (
    fetch_and_lock_tasks,
    get_connection,
    init_db,
    update_generation_failure,
    update_generation_success,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fake_vllm_generate(model_name: str, prompt: str, reasoning: bool) -> str:
    """Placeholder for vLLM generation.

    The actual system would query a model server.  For testing purposes we
    simply echo the prompt.
    """
    prefix = "REASONING" if reasoning else "BASE"
    return f"[{prefix}][{model_name}] {prompt}"


def process_batch(conn, batch_size: int) -> int:
    tasks = fetch_and_lock_tasks(conn, "PENDING_GENERATION", "GENERATING", batch_size)
    if not tasks:
        return 0

    for row in tasks:
        task_id = row["task_id"]
        model_name = row["model_name"]
        prompt = row["prompt"]
        try:
            base = fake_vllm_generate(model_name, prompt, reasoning=False)
            reasoning_resp = fake_vllm_generate(model_name, prompt, reasoning=True)
            update_generation_success(conn, task_id, base, reasoning_resp)
        except Exception as e:  # pragma: no cover - placeholder for real errors
            update_generation_failure(conn, task_id, str(e))
    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation worker")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of tasks per batch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = get_connection(cfg["database"])
    init_db(conn)

    processed = 0
    while True:
        count = process_batch(conn, args.batch_size)
        if count == 0:
            break
        processed += count
        # Sleep briefly to emulate polling behaviour
        time.sleep(0.1)

    print(f"generation_worker: processed {processed} tasks")


if __name__ == "__main__":
    main()
