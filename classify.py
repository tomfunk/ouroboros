"""Bucket a routing input (e.g. a sub-agent task description) with Jev Choice."""
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


def _api_key():
    if key := os.environ.get("TYPESAFE_API_KEY"):
        return key
    for line in (Path(__file__).parent / ".env").read_text().splitlines():
        name, _, value = line.removeprefix("export ").partition("=")
        if name.strip() == "TYPESAFE_API_KEY":
            return value.strip().strip("'\"")
    raise RuntimeError("TYPESAFE_API_KEY not set")


def classify(task):
    """Return (bucket, confidence, probabilities) for a task description."""
    body = {
        "state": task,
        "model": "jev-latest",
        "questions": {
            "bucket": {
                "type": "choice",
                "instructions": "Which kind of software engineering task is this",
                "criteria": BUCKETS,
            }
        },
    }
    req = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        a = json.load(resp)["answers"]["bucket"]
    return a["choice"], a["confidence"], a["probabilities"]


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
