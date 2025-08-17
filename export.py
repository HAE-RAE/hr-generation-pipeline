"""Export completed tasks to CSV or Parquet."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict

import yaml

try:  # pragma: no cover - pandas is optional for CSV export
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from task_db import get_connection


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def export_csv(rows, path: str) -> None:
    fieldnames = rows[0].keys() if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))


def export_parquet(rows, path: str) -> None:
    if pd is None:
        raise RuntimeError("pandas is required for parquet export")
    df = pd.DataFrame([dict(r) for r in rows])
    df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export results from task DB")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--format", choices=["csv", "parquet"], default="csv", help="Export format"
    )
    parser.add_argument(
        "--output", default="results/exported_tasks", help="Output file path without extension"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = get_connection(cfg["database"])

    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks WHERE status='COMPLETE'")
    rows = cur.fetchall()

    dirpath = os.path.dirname(args.output)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    out_path = f"{args.output}.{args.format}"

    if args.format == "csv":
        export_csv(rows, out_path)
    else:
        export_parquet(rows, out_path)

    print(f"Exported {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
