# ouroboros

Data-driven model routing without fine-tuning a router. The system's own routing history is the data used to tune it.

## Idea

Different models are good enough for different tasks. Cheap, fast models handle boilerplate. Expensive models handle hard reasoning. Asking a model "how hard is this?" is a guess with no evidence behind it. A fine-tuned router has evidence, but it goes stale: new models ship, model behavior shifts, and what works differs across orgs and repos.

So instead:

```
input → classify into a bucket (Jev Choice)
      → Thompson-sample a model for that bucket (with recency decay and a cost penalty)
      → collect the outcome (Jev Score + Noul judge, human feedback)
      → update that bucket's posterior for that model
      → periodically re-cut the buckets by editing the classifier prompt, then replay the history
```

## When to route

Switch models only where the prompt cache is already cold:

1. **Sub-agent spawn.** A new session, with a scoped task written by the parent. This is the best input for routing and the cleanest outcome signal. Start here.
2. **Post-compaction.** The compaction summary is a good routing input.
3. **Session start.** The weakest of the three, because the first prompts are often vague.

Switching late in a session costs the cached prefix again, so it is out of scope. One possible exception for later: switching up to a stronger model on strong failure signals, such as repeated test failures or repeated user corrections.

## The loop, step by step

Take one sub-agent spawn. The parent writes the task, for example "Write unit tests for the date parsing helper in utils/dates.py".

1. **Classify.** `jev.classify(task)` returns `("tests", 1.00, {...})`. The confidence is kept: a low value means the task sits between buckets.
2. **Route.** `router.route("tests")` draws θ from each model's Beta posterior for this bucket, subtracts `λ·cost`, and picks the highest score. A cheap model with a good track record usually wins. An uncertain model sometimes wins, and that is the exploration.
3. **Run.** The sub-agent runs on the chosen model, as normal. This is the one step this repo does not implement.
4. **Judge.** `jev.judge(task, response)` returns a 0–1 quality score from the rubric, plus three 0–1 Noul checks. `jev.reward()` averages them into one reward. Human feedback, such as "give me a smarter model", can override it.
5. **Update.** `router.update("tests", model, reward)` decays every model in the bucket a little toward the prior, then adds the reward to the chosen model. The next "tests" task is routed with this new evidence.
6. **Log.** `log_event(log, task=..., bucket=..., model=..., cost=..., judge=..., reward=...)` appends the raw record.
7. **Feed the log back in.** This is the ouroboros part. Every so often, read the log:
   - A bucket where no model wins, or where confidence is low, is a candidate to split. Edit `BUCKETS`.
   - `rebuild(log, bucket_fn=jev.classify)` re-buckets all the history with the new prompt and rebuilds the posteriors. No training, and no data lost.
   - `rebuild(log, reward_fn=...)` tries a different reward weighting on the old judge answers.
   - A new model is added with `router.add_model()`, and exploration finds out where it is good enough.

In code, steps 1–6 for one task:

```python
bucket, confidence, _ = jev.classify(task)
model = router.route(bucket)
response = run_subagent(model, task)    # your harness
j = jev.judge(task, response)
router.update(bucket, model, jev.reward(j))
log_event("log.jsonl", task=task, bucket=bucket, confidence=confidence,
          model=model, cost=router.costs[model], judge=j, reward=jev.reward(j))
```

`python3 router.py` runs this loop with simulated buckets and rewards. `python3 jev.py` runs steps 1 and 4 for real.

## The buckets don't need to mean anything

What a bucket is doesn't matter. "Tests", "debugging" and "design" are just convenient starting labels. A bucket is useful if it has two properties:

- **Coherent.** Tasks in the same bucket behave alike across models: the model that is good enough for one is good enough for the others.
- **Reachable every time.** The same kind of input lands in the same bucket every time. Jev makes this possible, because Choice returns a typed answer from a fixed list rather than free text, and its `confidence` shows when an input is close to a boundary.

The router never needs to know *why* a model suits a bucket. The statistics show empirically what works there, so nobody has to theorize about it. If the buckets were "tasks that mention dates" and "tasks that don't", the bandit would still learn the right model for each, as long as those groups behave consistently. That is why buckets can be cheap to change: edit the prompt, replay the log, and keep the new cut if the statistics get sharper.

## Pieces

- **Buckets** come from a [Jev](https://docs.typesafe.ai/introduction) Choice prompt. Cold start: take about 10 buckets and cut a sample of past conversations with them. Tune the buckets by editing the prompt, not by training. Choice `confidence` flags inputs that fall between buckets.
- **Bandit.** [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling) with a Beta posterior for each (bucket, model) pair. The router picks the model with the highest `θ − λ·cost`. λ is one knob that means "how much quality per dollar we demand". Every update decays all arms in the bucket toward the prior, so recent outcomes count more and models we stopped using get re-explored.
- **Reward** is a value in [0, 1]: a Jev Score rubric for explicit partial credit, plus Noul checks, combined in code, plus human feedback ("give me a smarter model"). Fractional updates: `α += r`, `β += 1 − r`.
- **New models** join with the uniform prior and get explored. Optionally, replay history with the new model to set an informed prior.
- **Raw log first.** Store each input, model, cost, and outcome. Buckets are a view on top of that log. When the classifier prompt changes, re-classify the history and rebuild the posteriors. The same log lets you test a new bucketing offline before you ship it.

## Open questions

- **Reward bias.** Human feedback is mostly negative and comes mostly from cheap-model routes. Users who say nothing did not necessarily succeed. LLM judges tend to favor long answers and possibly their own model family. Use objective signals where they exist: tests pass, the parent agent accepts the sub-agent's result.
- **Counterfactual replay** works for self-contained sub-agent tasks. It does not work honestly for multi-turn sessions, because the conversation goes differently once a different model answers.
- **Exploration costs real users.** Consider a minimum model tier for each bucket and a cap on exploration in high-stakes buckets.
- **Bucket coherence.** Two warning signs: no model wins despite lots of data, low Choice confidence, or both. Test: split the bucket's history by a sub-feature and check whether the best model changes between the halves.
- **Scope.** Posteriors per org or per repo, and how to share priors between them.

## Code

- `jev.py`: `classify(task)` uses Jev Choice over `BUCKETS`. `judge(task, response)` asks a Jev Score rubric (level index ÷ (levels − 1), so 0–1) plus the Noul `CHECKS`, and `reward()` averages them. `python3 jev.py` makes live calls and needs `TYPESAFE_API_KEY` in `.env`.
- `router.py`: `Router` is the bandit. `log_event()` appends raw JSONL. `rebuild(log, bucket_fn=, reward_fn=)` replays the log, with optional re-bucketing or re-weighting. `python3 router.py` runs offline and checks two things:
  - The simulation converges, adapts to drift, and picks up a new model.
  - A router rebuilt from its own log matches the live router exactly.

  It then prints the posterior grid: one row per bucket, one column per model. Each cell shows that Beta distribution (x = P(success), 0 to 1), its mean, its effective sample size, and win%, which is how often the router would pick that model once cost is included. `python3 router.py log.jsonl` prints the same grid for a real log.

  ```
               haiku (cost 0.05)     sonnet (cost 0.15)    opus (cost 0.40)      fable (cost 0.10)
               0··················1  0··················1  0··················1  0··················1
  boilerplate  ▁▂▃▄▅▆▇█████▇▇▆▅▄▃▁   ▁▂▃▄▄▅▆▆▇▇▇██████▇▆▄  ████████████████████                   ▁█▄
               μ0.49 n3 win3%        μ0.57 n1 win1%        μ0.50 n0 win0%        μ0.94 n96 win97%
  hard         ▂▄▅▆▇██████▇▇▆▆▅▄▃▂▁   ▁▁▂▃▄▄▅▅▆▇▇▇████▇▆▅  ████████████████████                   ▃█
               μ0.46 n2 win4%        μ0.62 n2 win3%        μ0.50 n0 win0%        μ0.91 n97 win93%
  ```

  A sharp spike means the router is confident. A flat bar is the prior: either a model was never tried, or decay has pulled it back so it gets re-explored. Each curve is scaled to its own peak, so read the certainty from `n`.

Not included, on purpose: running the routed model (step 3). That belongs to whatever harness spawns the sub-agents. Logs (`*.jsonl`) are git-ignored, because they contain task text.
