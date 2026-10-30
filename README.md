# ouroboros

Data-driven model routing without fine-tuning a router. It tunes itself on its own routing history.

This is a proof of concept. It classifies and judges with real [Jev](https://docs.typesafe.ai/introduction) calls and simulates everything else. It does not run the routed models: that belongs to whatever harness spawns the sub-agents.

```
python3 router.py             # offline: simulated loop, self-checks, posterior grid
python3 jev.py                # live: classify + judge examples (needs TYPESAFE_API_KEY in .env)
python3 router.py log.jsonl   # posterior grid for a real log
```

## Idea

Different models are good enough for different tasks. Cheap, fast models handle boilerplate. Expensive models handle hard reasoning. Asking a model "how hard is this?" is a guess with no evidence behind it. A fine-tuned router has evidence, but it goes stale: new models ship, model behavior shifts, and what works differs across orgs and repos.

So instead:

```
input → classify into a bucket (Jev Choice)
      → Thompson-sample a model for that bucket (with recency decay and a cost penalty)
      → judge the outcome (Jev Score + Noul checks, human feedback)
      → update that bucket's posterior for that model
      → periodically re-cut the buckets by editing the classifier prompt, then replay the history
```

Jev is TypeSafe's System One model. It doesn't generate text: it answers typed questions about an input with structured values. Choice picks one option from a list, Score rates against an ordinal rubric, and Noul tests a statement and returns 0–1. [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling) keeps a Beta posterior per (bucket, model) and picks by sampling from them, so it explores exactly as much as it is uncertain.

## The buckets don't need to mean anything

"Tests", "debugging" and "design" are just convenient starting labels. A bucket is useful if it is:

- **Coherent.** Tasks in the same bucket behave alike across models: the model that is good enough for one is good enough for the others.
- **Repeatable.** The same kind of input lands in the same bucket every time. Jev makes this possible, because Choice returns a typed answer from a fixed list, and its `confidence` shows when an input is close to a boundary.

The router never needs to know *why* a model suits a bucket. It just measures what works. Buckets of "tasks that mention dates" and "tasks that don't" would also work, as long as each group behaves consistently. So buckets are cheap to change. Start with about 10 and cut a sample of past conversations with them. A bucket where no model wins despite lots of data, or where confidence is often low, is a candidate to split. Edit the prompt, replay the log, and keep the new cut if the statistics get sharper.

## When to route

Switch models only where the prompt cache is already cold:

1. **Sub-agent spawn.** A new session, with a scoped task written by the parent. This is the best input for routing and the cleanest outcome signal. Start here.
2. **Post-compaction.** The compaction summary is a good routing input.
3. **Session start.** The weakest of the three, because the first prompts are often vague.

Switching late in a session costs the cached prefix again, so it is out of scope.

## The loop, step by step

Take one sub-agent spawn. The parent writes the task, for example "Write unit tests for the date parsing helper in utils/dates.py".

1. **Classify.** `jev.classify(task)` returns `("tests", 1.00, {...})`, the bucket plus Jev's confidence.
2. **Route.** `router.route("tests")` draws θ from each model's Beta posterior for this bucket and picks the highest `θ − λ·cost`. λ is one knob that means "how much quality per dollar we demand". A cheap model with a good track record usually wins. An uncertain model sometimes wins, and that is the exploration.
3. **Run.** The sub-agent runs on the chosen model, as normal. Your harness does this.
4. **Judge.** `jev.judge(task, response)` returns a 0–1 quality score from a fail/partial/full rubric, plus three 0–1 Noul checks. `jev.reward()` averages them into one reward `r`. Human feedback, such as "give me a smarter model", can override it.
5. **Update.** `router.update("tests", model, r)` first decays every model in the bucket a little toward the prior. Then it adds `α += r, β += 1 − r` to the chosen model. Decay makes recent outcomes count more, and models we stopped using get re-explored, so drift gets noticed.
6. **Log.** `log_event()` appends the raw record: task, bucket, confidence, model, cost, judge answers, reward. The log is the source of truth, and the posteriors are just a cache of it.
7. **Feed the log back in.** This is the ouroboros part. Every so often:
   - Edit `BUCKETS`, then `rebuild(log, bucket_fn=jev.classify)` re-buckets all the history and rebuilds the posteriors. No training, and no data lost. Do this offline to test a new cut before shipping it.
   - `rebuild(log, reward_fn=...)` tries a different reward weighting on the old judge answers.
   - `router.add_model()` adds a new model at the uniform prior, and exploration finds out where it is good enough.

In code, steps 1–6 for one task:

```python
bucket, confidence, _ = jev.classify(task)
model = router.route(bucket)
response = run_subagent(model, task)    # your harness
j = jev.judge(task, response)
r = jev.reward(j)
router.update(bucket, model, r)
log_event("log.jsonl", task=task, bucket=bucket, confidence=confidence,
          model=model, cost=router.costs[model], judge=j, reward=r)
```

## Seeing the stats

`python3 router.py` simulates this loop. It checks that the router converges, adapts to drift, picks up a new model, and rebuilds exactly from its own log. Then it prints one row per bucket and one column per model. Excerpt:

```
             haiku (cost 0.05)     fable (cost 0.10)
             0··················1  0··················1
boilerplate  ▁▂▃▄▅▆▇█████▇▇▆▅▄▃▁                   ▁█▄
             μ0.49 n3 win3%        μ0.94 n96 win97%
hard         ▂▄▅▆▇██████▇▇▆▆▅▄▃▂▁                   ▃█
             μ0.46 n2 win4%        μ0.91 n97 win93%
```

Each cell is that Beta posterior over P(success), with its mean, effective sample size, and win%, which is how often the router would pick that model once cost is included. A sharp spike means the router is confident. A wide curve means it isn't, because the model was never tried or decay has pulled it back toward the prior. Each curve is scaled to its own peak, so read the certainty from `n`.

## Open questions

- **Reward bias.** Human feedback is mostly negative and comes mostly from cheap-model routes. Users who say nothing did not necessarily succeed. Judges have biases too: LLM judges are known to favor long answers and their own model family, and Jev's biases on this task are untested. Use objective signals where they exist: tests pass, the parent agent accepts the sub-agent's result.
- **Counterfactual replay** works for self-contained sub-agent tasks. It could give a new model an informed prior instead of a uniform one. It does not work honestly for multi-turn sessions, because the conversation goes differently once a different model answers.
- **Exploration costs real users.** Informed priors per bucket could keep weak models away from high-stakes buckets, instead of adding a separate tier floor or exploration cap.
- **Bucket splits.** Is the split test reliable? Split a suspect bucket's history by a sub-feature and check whether the best model changes between the halves.
- **Switching up late in a session** on strong failure signals, such as repeated test failures or user corrections, might be worth the cache cost.
- **Scope.** Posteriors per org or per repo, and how to share priors between them.

Logs (`*.jsonl`) are git-ignored, because they contain task text.
