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
from typing import Any, Dict, List

import yaml

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - datasets is optional during tests
    load_dataset = None

from task_db import get_connection, init_db, insert_task


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataset_from_config(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load dataset records with question/gold/category fields."""
    dataset_cfg = config.get("dataset", {})
    records: List[Dict[str, str]] = []

    # Inline records for quick tests
    if isinstance(dataset_cfg.get("records"), list) and dataset_cfg["records"]:
        for r in dataset_cfg["records"]:
            if isinstance(r, dict):
                records.append(
                    {
                        "question": str(r.get("question", "")),
                        "gold": str(r.get("gold", "")),
                        "category": str(r.get("category", "")),
                    }
                )
    # Hugging Face dataset loader
    elif load_dataset is not None and dataset_cfg.get("name"):
        ds = load_dataset(dataset_cfg["name"], split=dataset_cfg.get("split", "train"))
        for ex in ds:
            records.append(
                {
                    "question": str(ex.get("question", "")),
                    "gold": str(ex.get("gold", "")),
                    "category": str(ex.get("category", "")),
                }
            )
    else:
        # Fallback sample to keep pipeline functional offline
        records = [
            {
                "question": "What is 1+1?\nA. 1\nB. 2",
                "gold": "B",
                "category": "mcqa",
            }
        ]

    sample_size = dataset_cfg.get("sample_size")
    if sample_size:
        records = records[: int(sample_size)]
    return records


def _prompt_for_category(rec: Dict[str, str]) -> str:
    """Create the final prompt based on dataset category."""
    category = (rec.get("category") or "").lower()
    q = rec.get("question", "")
    if "mcqa" in category or category.startswith("mcq"):
        return f"{q}\nAnswer with the letter of the correct option."
    if "math" in category:
        return f"{q}\nProvide only the final answer."
    return q


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up tasks in the database and exit"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Database initialisation
    conn = get_connection(config["database"])
    init_db(conn)

    # Load dataset records
    records = _load_dataset_from_config(config)

    # Insert tasks for every (question, model) combination
    for rec in records:
        prompt = _prompt_for_category(rec)
        for model in config.get("models", []):
            key = f"{prompt}\0{model['name']}"
            task_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
            insert_task(
                conn,
                task_id,
                prompt,
                rec.get("gold", ""),
                rec.get("category", ""),
                model["name"],
            )

    print(
        "Task setup complete: workers can now be started separately to process the tasks."
    )


if __name__ == "__main__":
    main()
