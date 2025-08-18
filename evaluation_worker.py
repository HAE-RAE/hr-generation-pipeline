"""Evaluation worker for Reasoning Model Evaluation Pipeline.

This worker consumes tasks that already have model responses generated and
produces scores and feedback for each response.  The evaluation logic is a
lightweight heuristic placeholder; in a production environment this would
call an external judge model such as GPT-4.
"""

from __future__ import annotations

import argparse
import re
import time
from typing import Any, Dict, List

import yaml

from task_db import (
    fetch_and_lock_tasks,
    get_connection,
    init_db,
    update_evaluation_failure,
    update_evaluation_success,
)

try:  # Optional dependency used for strict math checking
    from math_verify import parse as mv_parse, verify as mv_verify
    MATH_VERIFY_AVAILABLE = True
except Exception:  # pragma: no cover - best effort import
    MATH_VERIFY_AVAILABLE = False


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rubric(cfg: Dict[str, Any]) -> str:
    """Load evaluation rubric from a YAML file if provided."""
    if not cfg or "rubric_path" not in cfg:
        return ""
    with open(cfg["rubric_path"], "r", encoding="utf-8") as f:
        rubric_data = yaml.safe_load(f) or {}
    return rubric_data.get("rubric", "")


def load_prompt_template(cfg: Dict[str, Any]) -> str:
    """Load comparative judge prompt template if provided."""
    if not cfg or "prompt_template_path" not in cfg:
        return ""
    with open(cfg["prompt_template_path"], "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("prompt", "")


# ---------------------------------------------------------------------------
# Evaluation modules
# ---------------------------------------------------------------------------


def _parse_math_answer(response: str) -> str:
    """Extract the boxed answer or last number from a math response."""
    match = re.search(r"\\boxed\{(.*?)\}", response)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+", response)
    return numbers[-1] if numbers else ""


def _normalize_mqa_choice_advanced(response: str) -> str:
    """Normalize various multiple-choice formats into a single letter."""
    if not isinstance(response, str):
        return ""
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    match = re.search(r"정답은\s*([A-Ja-j1-5])\s*(?:입니다|이다)", response)
    if match:
        return match.group(1).upper().strip(".)")
    choices = re.findall(r"[A-Ea-e1-5]", response)
    if choices:
        last = choices[-1]
        if last.isdigit():
            return chr(ord("A") + int(last) - 1)
        return last.upper()
    return response.strip().upper().strip(".)")

def evaluate_llm_judge_individual(response: str, rubric: str) -> Dict[str, Any]:
    """Placeholder individual LLM judge based on response length."""
    score = min(len(response) // 10, 10)
    feedback = f"{rubric.strip()} | length={len(response)}" if rubric else f"length={len(response)}"
    return {"score": score, "feedback": feedback}


def evaluate_mcqa(response: str, gold: str) -> bool:
    """Advanced MCQA evaluator using robust choice parsing."""
    if not gold:
        return False
    pred = _normalize_mqa_choice_advanced(response)
    return pred == gold.strip().upper().strip(".)")


def evaluate_math(response: str, gold: str) -> bool:
    """Verify math answers using math_verify when available."""
    if not gold:
        return False
    if MATH_VERIFY_AVAILABLE:
        try:
            parsed_gold = mv_parse(str(gold))
            parsed_resp = mv_parse(response)
            return bool(mv_verify(parsed_gold, parsed_resp))
        except Exception:
            return False
    pred = _parse_math_answer(response)
    return pred == str(gold).strip()


def evaluate_llm_judge_comparative(
    base_resp: str, reasoning_resp: str, template: str | None = None
) -> Dict[str, str]:
    """Placeholder comparative judge choosing the longer response.

    The template argument is accepted for compatibility with future LLM-based
    judging but is unused in this heuristic implementation.
    """
    if len(base_resp) > len(reasoning_resp):
        choice = "base"
    elif len(base_resp) < len(reasoning_resp):
        choice = "reasoning"
    else:
        choice = "tie"
    reasoning = f"base_len={len(base_resp)}, reasoning_len={len(reasoning_resp)}"
    return {"choice": choice, "reasoning": reasoning}


def process_batch(
    conn, batch_size: int, evaluations: List[str], rubric: str, comp_prompt: str
) -> int:
    tasks = fetch_and_lock_tasks(conn, "GENERATION_COMPLETE", "EVALUATING", batch_size)
    if not tasks:
        return 0

    for row in tasks:
        task_id = row["task_id"]
        try:
            base_resp = row["base_response"] or ""
            reasoning_resp = row["reasoning_response"] or ""

            # Individual judge scores
            if "llm_judge_individual" in evaluations:
                base_eval = evaluate_llm_judge_individual(base_resp, rubric)
                reasoning_eval = evaluate_llm_judge_individual(reasoning_resp, rubric)
                base_score = base_eval["score"]
                reasoning_score = reasoning_eval["score"]
                base_feedback = base_eval["feedback"]
                reasoning_feedback = reasoning_eval["feedback"]
            else:
                base_score = reasoning_score = 0
                base_feedback = reasoning_feedback = ""

            # Objective correctness checks
            base_is_correct = reasoning_is_correct = None
            if "mcqa" in evaluations and row["gold"]:
                base_is_correct = int(evaluate_mcqa(base_resp, row["gold"]))
                reasoning_is_correct = int(evaluate_mcqa(reasoning_resp, row["gold"]))
            elif "math_verify" in evaluations and row["gold"]:
                base_is_correct = int(evaluate_math(base_resp, row["gold"]))
                reasoning_is_correct = int(evaluate_math(reasoning_resp, row["gold"]))

            # Comparative judge
            judge_choice = judge_reasoning = None
            if "llm_judge_comparative" in evaluations:
                comp = evaluate_llm_judge_comparative(base_resp, reasoning_resp, comp_prompt)
                judge_choice = comp["choice"]
                judge_reasoning = comp["reasoning"]

            # Determine choice based on scores
            if base_score > reasoning_score:
                choice = "base"
            elif base_score < reasoning_score:
                choice = "reasoning"
            else:
                choice = "tie"

            update_evaluation_success(
                conn,
                task_id,
                base_score,
                reasoning_score,
                base_feedback,
                reasoning_feedback,
                choice,
                base_is_correct,
                reasoning_is_correct,
                judge_choice,
                judge_reasoning,
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

    ew_cfg = cfg.get("evaluation_worker", {})
    evaluations = ew_cfg.get("evaluations_to_run", [])
    rubric = load_rubric(ew_cfg.get("individual_judge", {})) if "llm_judge_individual" in evaluations else ""
    comp_prompt = (
        load_prompt_template(ew_cfg.get("comparative_judge", {}))
        if "llm_judge_comparative" in evaluations
        else ""
    )

    processed = 0
    while True:
        count = process_batch(conn, args.batch_size, evaluations, rubric, comp_prompt)
        if count == 0:
            break
        processed += count
        time.sleep(0.1)

    print(f"evaluation_worker: processed {processed} tasks")


if __name__ == "__main__":
    main()
