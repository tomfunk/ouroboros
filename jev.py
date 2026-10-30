"""Jev calls: bucket a routing input (Choice) and judge a response (Noul)."""
import json
import os
import sys
import urllib.request
from pathlib import Path

# ponytail: hand-written starter buckets; tune by editing these, then replay the log.
BUCKETS = {
    "boilerplate": "Writing repetitive or templated code: scaffolding, CRUD, config, simple glue",
    "search": "Finding where something lives in a codebase; read-only lookup, no changes",
    "small_edit": "A small, well-specified change to one or two files",
    "tests": "Writing or fixing tests for existing behavior",
    "debugging": "Diagnosing a bug or failure whose cause is not yet known",
    "refactor": "Restructuring existing code across several files without changing behavior",
    "design": "Architecture, API or system design decisions with trade-offs",
    "review": "Reviewing code or a diff for correctness, security or quality",
    "docs": "Writing or editing prose: docs, READMEs, comments, messages",
    "research": "Open-ended investigation needing external info or many sources",
}

# Atomic pass/fail checks, combined in code (see reward()). Edit freely; the raw
# answers are logged, so old records can be re-weighted on replay.
CHECKS = {
    "correct": "The response is technically correct",
    "complete": "The response fully does what the task asked",
    "on_scope": "The response sticks to the task without unrequested extra work",
}


def _api_key():
    if key := os.environ.get("TYPESAFE_API_KEY"):
        return key
    for line in (Path(__file__).parent / ".env").read_text().splitlines():
        name, _, value = line.removeprefix("export ").partition("=")
        if name.strip() == "TYPESAFE_API_KEY":
            return value.strip().strip("'\"")
    raise RuntimeError("TYPESAFE_API_KEY not set")


def _ask(state, questions):
    req = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=json.dumps({"state": state, "model": "jev-latest", "questions": questions}).encode(),
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["answers"]


def classify(task):
    """Return (bucket, confidence, probabilities) for a task description."""
    a = _ask(task, {"bucket": {
        "type": "choice",
        "instructions": "Which kind of software engineering task is this",
        "criteria": BUCKETS,
    }})["bucket"]
    return a["choice"], a["confidence"], a["probabilities"]


def judge(task, response):
    """Return raw judge answers: {<check>: noul in [0, 1]}."""
    questions = {k: {"type": "noul", "instructions": v} for k, v in CHECKS.items()}
    answers = _ask(json.dumps({"task": task, "response": response}), questions)
    return {k: a["noul"] for k, a in answers.items()}


def reward(j):
    # ponytail: equal-weight mean; tune weights against human feedback once we have some.
    return sum(j.values()) / len(j)


if __name__ == "__main__":
    samples = sys.argv[1:] or [
        "Find where the auth middleware validates JWTs and report the file paths.",
        "Add a created_at column to the users model and a migration for it.",
        "Payments intermittently double-charge under load; figure out why.",
        "Write unit tests for the date parsing helper in utils/dates.py.",
        "Should we move from REST polling to websockets for live updates? Weigh it up.",
    ]
    for s in samples:
        bucket, conf, _ = classify(s)
        print(f"{bucket:12} {conf:.2f}  {s}")

    task = "Write a Python function that returns the n-th Fibonacci number, with fib(0) == 0."
    good = "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"
    bad = "def fib(n):\n    return n * fib(n - 1)  # also added a logging framework and a CLI"
    jg, jb = judge(task, good), judge(task, bad)
    print("good:", jg, f"reward={reward(jg):.2f}")
    print("bad: ", jb, f"reward={reward(jb):.2f}")
    assert all(0 <= v <= 1 for v in [*jg.values(), *jb.values()])
    assert reward(jg) > reward(jb) + 0.3, "judge failed to separate good from bad"
    print("ok")
