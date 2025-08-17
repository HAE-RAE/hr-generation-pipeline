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
from functools import lru_cache

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
    prefix = "REASONING" if reasoning else "BASE"
    return f"[{prefix}][{model_name}] {prompt}"


@lru_cache(maxsize=4)
def get_generator(model_id: str):
    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        return None
    try:
        gen = pipeline("text-generation", model=model_id, device=-1)
        return gen
    except Exception:
        return None


def generate_text(model_id: str, prompt: str, reasoning: bool, max_new_tokens: int) -> str:
    gen = get_generator(model_id)
    if gen is None:
        # Fallback to echo generation
        return fake_vllm_generate(model_id, prompt, reasoning)
    # Add a tiny reasoning cue to differentiate paths
    eff_prompt = prompt + ("\nLet's think step by step." if reasoning else "")
    out = gen(eff_prompt, max_new_tokens=max_new_tokens, do_sample=False)
    text = out[0].get("generated_text", "") if isinstance(out, list) and out else str(out)
    return text


def process_batch(conn, batch_size: int, model_map: Dict[str, str], max_new_tokens: int) -> int:
    tasks = fetch_and_lock_tasks(conn, "PENDING_GENERATION", "GENERATING", batch_size)
    if not tasks:
        return 0

    for row in tasks:
        task_id = row["task_id"]
        model_name = row["model_name"]
        prompt = row["prompt"]
        try:
            model_id = model_map.get(model_name, model_name)
            base = generate_text(model_id, prompt, reasoning=False, max_new_tokens=max_new_tokens)
            reasoning_resp = generate_text(
                model_id, prompt, reasoning=True, max_new_tokens=max_new_tokens
            )
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

    # Build model name -> HF model id mapping
    models_cfg = cfg.get("models", [])
    model_map = {m.get("name", ""): m.get("path", m.get("name", "")) for m in models_cfg if m.get("name")}

    sampling = cfg.get("generation_worker", {}).get("sampling_params", {})
    max_new_tokens = int(sampling.get("max_tokens", 32))
    # Keep it small for CPU tests
    max_new_tokens = max(1, min(max_new_tokens, 64))

    processed = 0
    while True:
        count = process_batch(conn, args.batch_size, model_map, max_new_tokens)
        if count == 0:
            break
        processed += count
        time.sleep(0.1)

    print(f"generation_worker: processed {processed} tasks")


if __name__ == "__main__":
    main()
