"""Toy bucketed Thompson-sampling model router.

Buckets come from a classifier (Jev Choice in the real thing); rewards in [0, 1]
come from a judge (Jev Score) or human feedback. Here both are simulated.
"""
import random


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


if __name__ == "__main__":
    simulate()
