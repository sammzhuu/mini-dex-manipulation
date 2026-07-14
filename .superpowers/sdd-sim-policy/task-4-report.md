# Task 4 Report — Evaluation + rollout export (sim_policy lane)

## What was implemented

1. `sim_policy/.gitignore` — single line `artifacts/` (matches brief exactly).
2. `sim_policy/tests/test_rollout_schema.py` — written exactly as specified in the brief:
   - `test_fixture_matches_schema`: validates `shared/fixtures/rollout.json` against `shared/schema/rollout_schema.json`.
   - `test_generated_rollout_matches_schema`: skips if `sim_policy/artifacts/rollout.json` doesn't exist yet; otherwise validates it against the schema.
3. `sim_policy/evaluate.py` — based on the brief's example code, with the required deviation applied at the object-pose extraction site. Instead of the brief's:
   ```python
   "position": qpos[-7:-4].tolist(),
   "orientation": qpos[-4:].tolist(),
   ```
   the script now uses MuJoCo body-level global pose:
   ```python
   # This installed gymnasium-robotics version's Adroit "Object" joint is a
   # 6-DOF OBJTx/Ty/Tz + OBJRx/Ry/Rz joint, not the classic 7-DOF quaternion
   # free joint — there is no 4-element quaternion in qpos to slice. Use the
   # body's global xpos/xquat instead, which is always a proper quaternion
   # regardless of the joint's own parameterization.
   obj_id = env.unwrapped.model.body("Object").id
   position = env.unwrapped.data.xpos[obj_id].tolist()
   orientation = env.unwrapped.data.xquat[obj_id].tolist()
   ```
   This is the only place in the codebase carrying this note, as instructed.

## TDD evidence

**RED** (before `evaluate.py` existed / before `rollout.json` existed — run right after writing the test file, step 3):
```
sim_policy/tests/test_rollout_schema.py::test_fixture_matches_schema PASSED [ 50%]
sim_policy/tests/test_rollout_schema.py::test_generated_rollout_matches_schema SKIPPED [100%]
1 passed, 1 skipped in 0.30s
```
This matches the brief's expected Step 3 output exactly — fixture already validates against the read-only schema, and the generated-rollout test correctly self-skips because `sim_policy/artifacts/rollout.json` did not exist yet.

**Run evaluate.py** (Step 5):
```
Wrote C:\Users\zjial\Documents\code\mini-dex-manipulation\sim_policy\artifacts\rollout.json
```
(One benign upstream gymnasium-robotics deprecation warning about `AdroitHand*Dense-v1` reward-function versioning printed to stderr first — unrelated to this task's env, `AdroitHandRelocateSparse-v1`, and not an error.)

Inspected `rollout.json`: `final_object_pose.orientation` values are well-formed 4-element quaternions (e.g. `[1.0, -4.2e-18, -2.3e-19, 1.5e-19]` for an episode where the object stayed near its reset pose — consistent with `xquat`'s `(w, x, y, z)` convention), confirming the `model.body("Object")` / `xpos` / `xquat` extraction resolved correctly and produced schema-valid shapes (3-elem position, 4-elem orientation).

**GREEN** (Step 6, after evaluate.py ran):
```
sim_policy/tests/test_rollout_schema.py::test_fixture_matches_schema PASSED [ 50%]
sim_policy/tests/test_rollout_schema.py::test_generated_rollout_matches_schema PASSED [100%]
2 passed in 0.09s
```

## Files changed / commit

- `sim_policy/.gitignore` (new)
- `sim_policy/evaluate.py` (new)
- `sim_policy/tests/test_rollout_schema.py` (new)

**Commit issue (important — read this):** `git add sim_policy/evaluate.py sim_policy/tests/test_rollout_schema.py sim_policy/.gitignore` succeeded and `git status` showed exactly those three files staged (`A`) with nothing else. But when I then ran `git commit -m "feat(sim_policy): add evaluation script and rollout schema test"`, git reported `nothing added to commit but untracked files present`.

Root cause: this repo is **not** using isolated worktrees per lane — both this sim_policy session and the concurrent perception-lane session share the same working tree and the same `.git/index`. In the window between my `git add` and my `git commit`, the perception-lane session ran its own commit (which evidently staged broadly, e.g. `git add -A`/`git commit -a`), and that commit swept up my three already-staged files along with its own perception changes. The result is that my three files landed in commit:

```
94981ae fix(perception): correct scipy Euler convention docs, add slicing regression test
 perception/capture.py                   | 15 ++++---
 perception/tests/test_capture_schema.py | 37 +++++++++++++++
 sim_policy/.gitignore                   |  1 +
 sim_policy/evaluate.py                  | 79 +++++++++++++++++++++++++++++++++
 sim_policy/tests/test_rollout_schema.py | 24 ++++++++++
```

I verified the **content** is exactly correct — `git show HEAD:sim_policy/evaluate.py`, `git show HEAD:sim_policy/tests/test_rollout_schema.py`, and `git show HEAD:sim_policy/.gitignore` are byte-identical to what I wrote (diffed against working tree, zero differences). So functionally the deliverable is committed and correct. What's wrong is provenance/hygiene: my sim_policy files are bundled into an unrelated perception commit with a perception-only commit message instead of getting their own `feat(sim_policy): ...` commit as the brief's Step 7 specifies.

I deliberately did **not** attempt to fix this via history rewriting (`git reset`, amend, rebase, cherry-pick-and-drop) because: (a) the global git-safety rules forbid destructive git operations unless explicitly requested, (b) the perception-lane session may still be actively committing to this same shared branch/index, and rewriting history out from under a concurrent process risks corruption or lost work that would be far worse than a mislabeled commit, (c) the content itself is verified correct and nothing is lost. This needs a decision from the project owner: either leave it as-is (content is fine, just an oddly-labeled commit), or coordinate a quiet moment to split it into a proper `feat(sim_policy): ...` commit.

## Self-review findings

- Both tests pass (confirmed twice, before and after evaluate.py ran).
- The extraction comment clearly explains the deviation (6-DOF Euler-style joint vs. 7-DOF quaternion free joint, why xpos/xquat is used instead of qpos slicing) and appears only once, at the exact extraction site in `evaluate.py`.
- `sim_policy/artifacts/` was correctly excluded from staging/commit — verified via `git status --porcelain sim_policy/` before commit (only `.gitignore`, `evaluate.py`, and `tests/` untracked/staged; `artifacts/` never appeared) and after (working tree clean apart from the unrelated `.superpowers/` dir).
- No scope creep: only the three specified files were created/modified. No edits to `perception/`, `ros2_bridge/`, `ops_dashboard/`, or `shared/`.

## Concerns

1. **Commit provenance** (see above): the three sim_policy files are correctly committed with correct content, but as part of commit `94981ae` ("fix(perception): ..."), not their own `feat(sim_policy): ...` commit, due to a shared-working-tree race with the concurrent perception-lane session. Flagging for project owner decision.
2. No functional/schema/logic concerns — the object-pose fix works exactly as described, both tests pass, and the generated rollout.json validates against the read-only schema.
