# Stretch task report: dense-reward retrain with VecNormalize

## What was implemented

Replaced the three files exactly as specified in `stretch-vecnorm-spec.md`:

1. `sim_policy/train.py` — switched `TASK_ID` from `AdroitHandRelocateSparse-v1`
   to `AdroitHandRelocate-v1` (dense reward), wrapped the vec env with
   `VecNormalize(vec_env, norm_obs=True, norm_reward=True)`, bumped the
   default `--timesteps` to `1_500_000`, and saved the normalizer stats to
   `artifacts/vecnormalize.pkl` alongside the model.
2. `sim_policy/evaluate.py` — switched `TASK_ID` to the dense variant, added
   a `vecnormalize_path` positional param to `evaluate()`, loads the saved
   `VecNormalize` stats onto a throwaway `DummyVecEnv`, sets
   `training=False` / `norm_reward=False`, and normalizes each observation
   via `vecnorm.normalize_obs(obs)` before `model.predict()`. Added the
   `--vecnormalize` CLI arg and updated the `evaluate()` call site in
   `__main__`. No change to the `success = bool(info.get("success", ...))`
   logic per the controller's confirmation that `info["success"]` is
   computed identically for both dense and sparse task IDs.
3. `sim_policy/record_demo.py` — same normalization pattern as
   `evaluate.py`: switched `TASK_ID`, added `vecnormalize_path` param to
   `record()`, loads stats via `DummyVecEnv` + `VecNormalize.load`, and
   normalizes observations before `model.predict()` in the render loop.

All three files were transcribed verbatim from the spec — no improvised
variants.

## Smoke-test commands and output

1. **Train (2000 timesteps):**
   ```
   sim_policy/.venv/Scripts/python.exe sim_policy/train.py --timesteps 2000
   ```
   Output (tail):
   ```
   Using cpu device
   ---------------------------------
   | rollout/           |          |
   |    ep_len_mean     | 200      |
   |    ep_rew_mean     | 13.6     |
   | time/              |          |
   |    fps             | 882      |
   |    iterations      | 1        |
   |    time_elapsed    | 9        |
   |    total_timesteps | 8192     |
   ---------------------------------
   Saved model to C:\Users\zjial\Documents\code\mini-dex-manipulation\sim_policy\artifacts\ppo_v1.zip
   ```
   Confirmed via `ls sim_policy/artifacts/`: both `ppo_v1.zip` (221,851
   bytes) and `vecnormalize.pkl` (4,654 bytes) exist with fresh timestamps.

   Note: PPO's default `n_steps=2048` * `n_envs=4` = 8192 minimum timesteps
   per rollout collection, so it ran one full rollout (8192 steps) rather
   than stopping exactly at 2000 — this is standard SB3 behavior (timesteps
   is a floor, not an exact cutoff) and not a bug.

2. **Evaluate (2 episodes):**
   ```
   sim_policy/.venv/Scripts/python.exe sim_policy/evaluate.py --episodes 2
   ```
   Output: `Wrote C:\Users\zjial\Documents\code\mini-dex-manipulation\sim_policy\artifacts\rollout.json`
   Completed without error — normalization wiring (`VecNormalize.load` +
   `normalize_obs`) worked correctly with the freshly-saved
   `vecnormalize.pkl` from step 1.

3. **Pytest schema tests:**
   ```
   sim_policy/.venv/Scripts/pytest.exe sim_policy/tests/test_rollout_schema.py -v
   ```
   Output:
   ```
   sim_policy/tests/test_rollout_schema.py::test_fixture_matches_schema PASSED [ 50%]
   sim_policy/tests/test_rollout_schema.py::test_generated_rollout_matches_schema PASSED [100%]
   2 passed in 0.12s
   ```

`record_demo.py` was intentionally not smoke-tested, per the spec's
instruction (shares the same normalization pattern already verified in
evaluate.py).

## Commit made

```
6a22cab feat(sim_policy): switch to dense reward + VecNormalize for retrain
 3 files changed, 47 insertions(+), 20 deletions(-)
```

Files: `sim_policy/train.py`, `sim_policy/evaluate.py`,
`sim_policy/record_demo.py` — exactly the 3 intended files, verified via
`git status --short` immediately before staging (only these 3 showed as
modified; `sim_policy/artifacts/` did not appear, confirming it's
gitignored as expected) and again immediately before committing (staged
list showed exactly the 3 files; only remaining untracked entry was
`.superpowers/`, which is outside `sim_policy/` and was left untouched).

## Self-review

- All 3 smoke-test commands pass: YES (train, evaluate, pytest all
  succeeded as shown above).
- Both `ppo_v1.zip` and `vecnormalize.pkl` exist after the smoke-test
  training run: YES, confirmed via directory listing with fresh
  timestamps.
- Exactly the 3 intended files staged before commit (verified via
  `git status --short`): YES.
- Code matches the spec's exact given code: YES, transcribed verbatim,
  including the explanatory comments in evaluate.py and record_demo.py.

## Concerns

None. The normalization smoke test round-tripped end to end (train saves
`vecnormalize.pkl` → evaluate loads it, normalizes observations, evaluates
without error) with no API mismatches on the installed
`stable-baselines3==2.9.0`. Left `ppo_v1.zip` / `vecnormalize.pkl` as the
2000-timestep smoke-test artifacts per instruction — the controller will
overwrite them with the real 1.5M-timestep run. Per the spec, the real long
training run was NOT started.
