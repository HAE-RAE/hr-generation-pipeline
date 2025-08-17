"""Orchestrator module for Reasoning Model Evaluation Pipeline.

This script reads a configuration file, creates the task database and
populates it with all combinations of prompts and models.  Each inserted
row represents a unit of work that will later be consumed by the
`generation_worker` and `evaluation_worker`.

The implementation intentionally focuses on clarity and debuggability.
It does not attempt to cover every optimisation from the design
specification but instead provides a solid reference implementation that
can be extended.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from typing import Any, Dict

import yaml

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - datasets is optional during tests
    load_dataset = None

from task_db import get_connection, init_db, insert_task


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prompts_from_config(config: Dict[str, Any]) -> list[str]:
    dataset_cfg = config.get("dataset", {})
    # 1) Inline prompts list
    if isinstance(dataset_cfg.get("prompts"), list) and dataset_cfg["prompts"]:
        prompts = [str(p) for p in dataset_cfg["prompts"]]
    # 2) Local text file (one prompt per line)
    elif isinstance(dataset_cfg.get("file"), str):
        path = dataset_cfg["file"]
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    # 3) Hugging Face datasets if available
    elif load_dataset is not None and dataset_cfg.get("name"):
        ds = load_dataset(dataset_cfg["name"], split=dataset_cfg.get("split", "train"))
        prompts = ds[dataset_cfg.get("prompt_column", "prompt")]
    else:
        # Fallback minimal prompts for offline usage
        prompts = ["Say hello.", "Summarize: AI helps humans."]

    sample_size = dataset_cfg.get("sample_size")
    if sample_size:
        prompts = prompts[: int(sample_size)]
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate experiment setup")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only set up tasks and exit (default behaviour)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Database initialisation
    conn = get_connection(config["database"])
    init_db(conn)

    # Load prompts (supports inline/file/HF datasets)
    prompts = _load_prompts_from_config(config)

    # Insert tasks for every (prompt, model) combination
    for prompt in prompts:
        for model in config.get("models", []):
            key = f"{prompt}\0{model['name']}"
            task_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
            insert_task(conn, task_id, prompt, model["name"])

    if args.setup_only:
        print("Task setup complete: tasks stored in DB")
        return

    print(
        "Setup complete. Workers can now be started separately to process the tasks."
    )


if __name__ == "__main__":
    main()
