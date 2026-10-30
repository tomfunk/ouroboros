"""Toy bucketed Thompson-sampling model router.

Buckets come from a classifier (Jev Choice in the real thing); rewards in [0, 1]
come from a judge (Jev Score) or human feedback. Here both are simulated.
"""
import json
import random
import time


def log_event(path, **record):
    """Append one routing outcome. Keep it raw (task, model, cost, judge answers)."""
    with open(path, "a") as f:
        f.write(json.dumps({"ts": time.time(), **record}) + "\n")


def rebuild(path, lam=1.0, gamma=0.99, reward_fn=None, bucket_fn=None):
    """Replay the log in order into a fresh Router.

    Pass bucket_fn (e.g. jev.classify) to re-bucket history after editing buckets,
    or reward_fn to re-weight judge answers. Otherwise uses what was logged.
    """
    r = Router({}, lam, gamma)
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e["model"] not in r.costs:
                r.add_model(e["model"], e["cost"])  # an arm at the prior decays as a no-op, so late add is exact
            bucket = bucket_fn(e["task"])[0] if bucket_fn else e["bucket"]
            r.update(bucket, e["model"], reward_fn(e["judge"]) if reward_fn else e["reward"])
    return r


class Router:
    def __init__(self, costs, lam=1.0, gamma=0.99):
        self.costs = dict(costs)  # model -> relative cost per call
        self.lam = lam            # quality we demand per unit cost
        self.gamma = gamma        # recency decay; ~1/(1-gamma) effective window
        self.post = {}            # (bucket, model) -> [alpha, beta]

    def add_model(self, model, cost):
        self.costs[model] = cost  # starts at the uniform prior, gets explored

    def _ab(self, bucket, model):
        return self.post.setdefault((bucket, model), [1.0, 1.0])

    def route(self, bucket):
        return max(self.costs, key=lambda m: random.betavariate(*self._ab(bucket, m)) - self.lam * self.costs[m])

    def update(self, bucket, model, reward):
        # Decay every arm in the bucket toward the prior, so arms we stopped
        # picking regain uncertainty and get re-explored (catches drift).
        for m in self.costs:
            ab = self._ab(bucket, m)
            ab[0] = 1 + self.gamma * (ab[0] - 1)
            ab[1] = 1 + self.gamma * (ab[1] - 1)
        ab = self._ab(bucket, model)
        ab[0] += reward
        ab[1] += 1 - reward


def simulate(steps=3000, seed=0):
    random.seed(seed)
    costs = {"haiku": 0.05, "sonnet": 0.15, "opus": 0.40}
    truth = {  # hidden P(success) per bucket per model
        "boilerplate": {"haiku": 0.90, "sonnet": 0.93, "opus": 0.95},
        "hard": {"haiku": 0.30, "sonnet": 0.60, "opus": 0.95},
    }
    r = Router(costs)

    def phase(n):
        picks = {b: [] for b in truth}
        for _ in range(n):
            b = random.choice(list(truth))
            m = r.route(b)
            r.update(b, m, float(random.random() < truth[b][m]))
            picks[b].append(m)
        return {b: max(set(p[-300:]), key=p[-300:].count) for b, p in picks.items()}, picks

    def share(picks, b, m):
        tail = picks[b][-300:]
        return tail.count(m) / len(tail)

    best, picks = phase(steps)
    print("phase 1 (converge):", best)
    assert best == {"boilerplate": "haiku", "hard": "opus"}
    assert share(picks, "boilerplate", "haiku") > 0.7

    truth["boilerplate"]["haiku"] = 0.50  # haiku "vibes" degrade
    best, picks = phase(steps)
    print("phase 2 (haiku drifts):", best)
    assert best["boilerplate"] == "sonnet"

    r.add_model("fable", 0.10)  # new model shows up, good at everything
    truth["boilerplate"]["fable"] = 0.95
    truth["hard"]["fable"] = 0.90
    best, picks = phase(steps)
    print("phase 3 (fable added):", best)
    assert best == {"boilerplate": "fable", "hard": "fable"}
    print("ok")


def check_replay(path):
    """Live router state must equal the state rebuilt from its own log."""
    random.seed(1)
    costs = {"haiku": 0.05, "sonnet": 0.15, "opus": 0.40}
    r = Router(costs)
    for i in range(500):
        if i == 250:
            r.add_model("fable", 0.10)
        b = random.choice(["boilerplate", "hard"])
        m = r.route(b)
        rew = random.random()
        r.update(b, m, rew)
        log_event(path, task=f"t{i}", bucket=b, model=m, cost=r.costs[m], judge={}, reward=rew)
    rr = rebuild(path)
    assert r.costs == rr.costs
    assert all(abs(a - c) < 1e-9 for k in r.post for a, c in zip(r.post[k], rr.post[k])), "replay diverged"
    print("replay ok")


if __name__ == "__main__":
    import os
    import tempfile

    simulate()
    with tempfile.TemporaryDirectory() as d:
        check_replay(os.path.join(d, "log.jsonl"))
