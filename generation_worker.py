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


@lru_cache(maxsize=2)
def get_vllm(model_id: str, **engine_params):
    try:
        from vllm import LLM  # type: ignore
    except Exception:
        return None
    # Filter known-safe params to avoid runtime errors
    allowed = {
        "tensor_parallel_size",
        "dtype",
        "gpu_memory_utilization",
        "trust_remote_code",
        "download_dir",
        "enforce_eager",
        "max_num_batched_tokens",
    }
    kwargs = {k: v for k, v in engine_params.items() if k in allowed}
    try:
        return LLM(model=model_id, **kwargs)
    except Exception:
        return None


def build_prompt(base_prompt: str, reasoning: bool, prompting: Dict[str, Any], control: Dict[str, Any]) -> str:
    strategy = (prompting or {}).get("strategy", "append")
    on_text = (prompting or {}).get("on_text", "Let's think step by step.")
    off_text = (prompting or {}).get("off_text", "")
    add = on_text if reasoning else off_text

    control_line = ""
    if isinstance(control, dict) and control.get("name"):
        name = str(control["name"]).strip()
        value = control.get("on_value") if reasoning else control.get("off_value")
        if name and value is not None:
            control_line = f"<{name}>{value}</{name}>"

    def join_lines(parts):
        return "\n".join([p for p in parts if p])

    if strategy == "prepend":
        return join_lines([control_line, add, base_prompt])
    # default: append
    return join_lines([base_prompt, add, control_line])


def generate_with_transformers(model_id: str, prompt: str, reasoning: bool, max_new_tokens: int, temperature: float, prompting: Dict[str, Any], control: Dict[str, Any]) -> str:
    gen = get_generator(model_id)
    if gen is None:
        return fake_vllm_generate(model_id, prompt, reasoning)
    eff_prompt = build_prompt(prompt, reasoning, prompting, control)
    out = gen(eff_prompt, max_new_tokens=max_new_tokens, do_sample=bool(temperature > 0), temperature=temperature)
    text = out[0].get("generated_text", "") if isinstance(out, list) and out else str(out)
    return text


def generate_with_vllm(model_id: str, prompt: str, reasoning: bool, max_new_tokens: int, temperature: float, prompting: Dict[str, Any], control: Dict[str, Any], engine_params: Dict[str, Any]) -> str:
    try:
        from vllm import SamplingParams  # type: ignore
    except Exception:
        return fake_vllm_generate(model_id, prompt, reasoning)
    llm = get_vllm(model_id, **(engine_params or {}))
    if llm is None:
        return fake_vllm_generate(model_id, prompt, reasoning)
    eff_prompt = build_prompt(prompt, reasoning, prompting, control)
    sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    try:
        outputs = llm.generate([eff_prompt], sp)
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
    except Exception:
        pass
    return fake_vllm_generate(model_id, prompt, reasoning)


def generate_text(backend: str, model_id: str, prompt: str, reasoning: bool, max_new_tokens: int, temperature: float, prompting: Dict[str, Any], control: Dict[str, Any], engine_params: Dict[str, Any]) -> str:
    backend = (backend or "transformers").lower()
    if backend == "vllm":
        return generate_with_vllm(model_id, prompt, reasoning, max_new_tokens, temperature, prompting, control, engine_params)
    return generate_with_transformers(model_id, prompt, reasoning, max_new_tokens, temperature, prompting, control)


def process_batch(
    conn,
    batch_size: int,
    backend: str,
    model_map: Dict[str, str],
    sampling: Dict[str, Any],
    prompting: Dict[str, Any],
    control: Dict[str, Any],
    engine_params: Dict[str, Any],
) -> int:
    tasks = fetch_and_lock_tasks(conn, "PENDING_GENERATION", "GENERATING", batch_size)
    if not tasks:
        return 0

    max_new_tokens = int(sampling.get("max_tokens", 32))
    max_new_tokens = max(1, min(max_new_tokens, 512))
    temperature = float(sampling.get("temperature", 0.0))

    for row in tasks:
        task_id = row["task_id"]
        model_name = row["model_name"]
        prompt = row["question"]
        try:
            model_id = model_map.get(model_name, model_name)
            base = generate_text(
                backend,
                model_id,
                prompt,
                reasoning=False,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompting=prompting,
                control=control,
                engine_params=engine_params,
            )
            reasoning_resp = generate_text(
                backend,
                model_id,
                prompt,
                reasoning=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompting=prompting,
                control=control,
                engine_params=engine_params,
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

    # Build model name -> model id mapping
    models_cfg = cfg.get("models", [])
    model_map = {m.get("name", ""): m.get("path", m.get("name", "")) for m in models_cfg if m.get("name")}

    gw_cfg = cfg.get("generation_worker", {})
    backend = gw_cfg.get("backend", "transformers")
    sampling = gw_cfg.get("sampling_params", {})
    prompting = gw_cfg.get("reasoning_prompting", {})
    control = gw_cfg.get("reasoning_control_param", {})
    engine_params = gw_cfg.get("vllm_engine_params", {})

    processed = 0
    while True:
        count = process_batch(
            conn,
            args.batch_size,
            backend,
            model_map,
            sampling,
            prompting,
            control,
            engine_params,
        )
        if count == 0:
            break
        processed += count
        time.sleep(0.1)

    print(f"generation_worker: processed {processed} tasks")


if __name__ == "__main__":
    main()
