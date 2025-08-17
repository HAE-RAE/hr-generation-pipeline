"""Evaluation worker for Reasoning Model Evaluation Pipeline.

This worker consumes tasks that already have model responses generated and
produces scores and feedback for each response.  The evaluation logic is a
lightweight heuristic placeholder; in a production environment this would
call an external judge model such as GPT-4.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Tuple

import yaml

from task_db import (
    fetch_and_lock_tasks,
    get_connection,
    init_db,
    update_evaluation_failure,
    update_evaluation_success,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(cfg: Dict[str, Any]) -> Tuple[str, str]:
    """Load judge prompt template and rubric from YAML files."""
    with open(cfg["judge_prompt_template_path"], "r", encoding="utf-8") as f:
        judge_data = yaml.safe_load(f) or {}
    with open(cfg["rubric_path"], "r", encoding="utf-8") as f:
        rubric_data = yaml.safe_load(f) or {}
    return judge_data.get("template", ""), rubric_data.get("rubric", "")


def simple_judge(response: str, rubric: str) -> Dict[str, Any]:
    """A naive evaluation heuristic.

    Scores responses based on length as a stand-in for a real LLM judge.
    The rubric text is appended to the feedback for demonstration purposes.
    """
    score = min(len(response) // 10, 10)  # 0-10 scale
    feedback = f"{rubric.strip()} | length={len(response)}" if rubric else f"length={len(response)}"
    return {"score": score, "feedback": feedback}


def process_batch(conn, batch_size: int, rubric: str) -> int:
    tasks = fetch_and_lock_tasks(
        conn, "GENERATION_COMPLETE", "EVALUATING", batch_size
    )
    if not tasks:
        return 0

    for row in tasks:
        task_id = row["task_id"]
        try:
            base_eval = simple_judge(row["base_response"] or "", rubric)
            reasoning_eval = simple_judge(row["reasoning_response"] or "", rubric)
            # Determine choice
            if base_eval["score"] > reasoning_eval["score"]:
                choice = "base"
            elif base_eval["score"] < reasoning_eval["score"]:
                choice = "reasoning"
            else:
                choice = "tie"
            update_evaluation_success(
                conn,
                task_id,
                base_eval["score"],
                reasoning_eval["score"],
                base_eval["feedback"],
                reasoning_eval["feedback"],
                choice,
            )
        except Exception as e:  # pragma: no cover
            update_evaluation_failure(conn, task_id, str(e))
    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation worker")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--batch-size", type=int, default=8, help="Tasks per batch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = get_connection(cfg["database"])
    init_db(conn)
    judge_template, rubric = load_prompts(cfg["evaluation_worker"])

    processed = 0
    while True:
        count = process_batch(conn, args.batch_size, rubric)
        if count == 0:
            break
        processed += count
        time.sleep(0.1)

    print(f"evaluation_worker: processed {processed} tasks")


if __name__ == "__main__":
    main()
