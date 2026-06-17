#!/usr/bin/env python3
"""Send sample prompts through the router dry-run API and print routing decisions.

Does not call upstream LLMs — only exercises feature extraction and model selection.

Usage:
    # Stack running via docker compose (API on host port 8001)
    python scripts/routing_probe.py

    # Custom base URL
    python scripts/routing_probe.py --base-url http://localhost:8001

    # Single prompt
    python scripts/routing_probe.py --prompt "Debug this race condition in the async handler"

Requires: httpx (`pip install httpx` or backend requirements)
"""

from __future__ import annotations

import argparse
import json
import sys

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx", file=sys.stderr)
    sys.exit(1)

SAMPLE_PROMPTS: list[tuple[str, list[dict]]] = [
    ("chitchat", [{"role": "user", "content": "Hello!"}]),
    ("summarization", [{
        "role": "user",
        "content": "Summarize this policy document in three bullet points.",
    }]),
    ("code_edit", [{
        "role": "user",
        "content": "Implement a Python function that reverses a linked list in place.",
    }]),
    ("debug", [{
        "role": "user",
        "content": (
            "Debug why this async handler intermittently drops messages. "
            "Find the race condition in these functions."
        ),
    }]),
    ("planning", [{
        "role": "user",
        "content": (
            "Design a migration plan from our monolith to microservices. "
            "Compare trade-offs and recommend an approach."
        ),
    }]),
    ("long_context", [{
        "role": "user",
        "content": " ".join(["word"] * 1500) + " Summarize the key themes.",
    }]),
    ("strict_requirements", [{
        "role": "user",
        "content": (
            "Implement OAuth2 refresh token rotation. Must preserve the public API. "
            "Output valid JSON schema. Include unit tests with pytest."
        ),
    }]),
    ("system_noise", [
        {
            "role": "system",
            "content": "You are helpful. Docs: https://example.com/rules",
        },
        {"role": "user", "content": "What is 2+2?"},
    ]),
]


def probe(base_url: str, messages: list[dict], label: str, verbose: bool = False) -> dict:
    url = f"{base_url.rstrip('/')}/api/v1/debug/complexity"
    resp = httpx.post(url, json={"messages": messages}, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", {})
    print(f"\n{'─' * 72}")
    print(f"  {label}")
    print(f"{'─' * 72}")
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    preview = user if isinstance(user, str) else str(user)
    if len(preview) > 120:
        preview = preview[:120] + "…"
    print(f"  Prompt: {preview}")
    print(f"  → Model:              {data.get('model_id')}")
    print(f"  → routing_method:     {data.get('routing_method')}")
    print(f"  → task_type:          {features.get('task_type')}")
    print(f"  → task_difficulty:    {features.get('task_difficulty')}")
    print(f"  → routing_difficulty: {data.get('routing_difficulty')}")
    print(f"  → context_load:       {features.get('context_load')}")
    print(f"  → requirement_load:   {features.get('requirement_load')}")
    if verbose:
        breakdown = data.get("complexity_explanation", {}).get("task_difficulty_breakdown", {})
        print(f"  → breakdown:          {json.dumps(breakdown)}")
    if features.get("embedding_routing_applied"):
        print(f"  → embedding:          {features.get('embedding_difficulty')} "
              f"(heuristic {features.get('heuristic_task_difficulty')})")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe LLM router with sample prompts")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8001",
        help="Router API base URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--prompt",
        help="Run a single custom user prompt instead of the built-in suite",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print task_difficulty breakdown JSON per prompt",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON responses",
    )
    args = parser.parse_args()

    # Health check
    try:
        health = httpx.get(f"{args.base_url.rstrip('/')}/health", timeout=5.0)
        health.raise_for_status()
    except Exception as exc:
        print(f"Cannot reach router at {args.base_url}: {exc}", file=sys.stderr)
        print("Start the stack: docker compose up -d", file=sys.stderr)
        sys.exit(1)

    models = httpx.get(f"{args.base_url.rstrip('/')}/api/v1/models", timeout=10.0)
    models.raise_for_status()
    model_count = models.json().get("total", 0)
    if model_count == 0:
        print("No models registered. Run: python scripts/seed_models.py", file=sys.stderr)
        sys.exit(1)

    print(f"Router OK — {model_count} models registered")
    print(f"API: {args.base_url}")

    cases: list[tuple[str, list[dict]]]
    if args.prompt:
        cases = [("custom", [{"role": "user", "content": args.prompt}])]
    else:
        cases = SAMPLE_PROMPTS

    results = []
    for label, messages in cases:
        data = probe(args.base_url, messages, label, verbose=args.verbose)
        results.append({"label": label, **data})
        if args.json:
            print(json.dumps(data, indent=2))

    print(f"\n{'═' * 72}")
    print("  Summary")
    print(f"{'═' * 72}")
    for r in results:
        f = r.get("features", {})
        print(
            f"  {r['label']:22s}  →  {r.get('model_id', '?'):25s}  "
            f"diff={f.get('task_difficulty', 0):.2f}  type={f.get('task_type', '?')}"
        )
    print()


if __name__ == "__main__":
    main()
