## sim_policy Lane Progress Ledger (namespaced to avoid collisions with other parallel lanes in shared .superpowers/sdd/)

Task 1: complete (commits b6c376f..ba3d0ed, review clean; deviation: deps repinned to gymnasium 1.3.0/gymnasium-robotics 1.4.2/mujoco 3.10.0/stable-baselines3 2.9.0/jsonschema 4.26.0/pytest 9.1.1 due to mujoco<3.0 no longer on PyPI, approved)

Task 2: complete (inspection only, no commit). Finding: installed gymnasium-robotics==1.4.2 AdroitHandRelocateSparse-v1 uses 6-DOF object joint (OBJTx/Ty/Tz + OBJRx/Ry/Rz Euler-like), not classic 7-DOF quaternion free joint. Approved convention for Task 4: env.model.body('Object').id then env.data.xpos[id] (3-elem pos) + env.data.xquat[id] (4-elem quat wxyz).

Task 3: in progress. train.py written and committed (8009cb7). Smoke test (2000 timesteps) passed. Real 200000-timestep training run in progress as of last check (16384/200000). Waiting for completion before proceeding to Task 4.
NOTE: .superpowers/sdd/task-3-report.md and .superpowers/sdd/progress.md are SHARED across parallel lanes and get overwritten by other lanes concurrent SDD runs -- do not trust their contents for this lane. Use this file (.superpowers/sdd-sim-policy/progress.md) instead, and verify actual state via `git log -- sim_policy/` and artifact file mtimes.

Task 3: complete (commit 8009cb7 "feat(sim_policy): add PPO training script"). Smoke test (2000 timesteps) passed. Real training run completed independently as an OS background process even after the monitoring subagent died from an API session limit (~03:01 local) -- verified via PPO.load(...).num_timesteps == 204800 (>=200000, confirms full run not smoke test). sim_policy/artifacts/ppo_v1.zip present, not committed (correct, per plan -- gitignore added in Task 4).
Task 3 review: clean, approved (commit a7a6513..8009cb7).

Task 4: complete. evaluate.py + test_rollout_schema.py + .gitignore written. Object-pose extraction uses env.unwrapped.model.body("Object").id + data.xpos/data.xquat (per Task 2 finding) instead of qpos slicing. Both schema tests PASS (verified independently by controller). rollout.json generated and validates.
DEVIATION HANDLED: implementer's git add+commit raced with a concurrent perception-lane session sharing the same working tree (no worktree isolation) -- sim_policy's 3 files got swept into perception's commit 94981ae. Controller split it via git reset --soft HEAD~1 (94981ae was still branch tip, safe) into two clean commits: 8327438 (perception, original message preserved) and aea561f (sim_policy, feat(sim_policy): add evaluation script and rollout schema test). User approved this approach before executing.
Task 4 review: clean, approved (commit 8327438..aea561f). Minor note only: unguarded div-by-zero on num_episodes=0, inherited from brief's own example code, low-risk, not fixed.

Task 5: complete (commit b3f88b1 "feat(sim_policy): add demo video recording script"). Implementer's session died from a connection error mid report-write, but the actual work completed first: commit is clean (only record_demo.py), sim_policy/artifacts/demo.mp4 verified independently by controller (201 frames, 6.7s @ 30fps, h264 480x480, not corrupt).
Task 5 review: clean, approved (commit b23ba70..b3f88b1).

ALL MVP TASKS (1-5) COMPLETE. Lane MVP checklist:
- sim_policy/artifacts/rollout.json exists, passes pytest sim_policy/tests/ (2/2 passed): YES
- sim_policy/artifacts/demo.mp4 exists, few seconds of visible motion (6.7s, 480x480, h264): YES
Next: final whole-branch review scoped to sim_policy/ commits, then finishing-a-development-branch.

FINAL WHOLE-BRANCH REVIEW: complete. Ready to merge: Yes. No Critical/Important findings.
4 Minor findings (none blocking, none fixed): (1) rollout loop duplicated across evaluate.py/record_demo.py -- reviewer recommends against extracting given YAGNI/deliberately-standalone scripts; (2) no try/finally around env in evaluate.py/record_demo.py, negligible leak risk in run-once CLI scripts; (3) rollout.json's video_path written unconditionally even if record_demo.py never run -- worth a README note at integration time; (4) already-known ZeroDivisionError on --episodes 0, inherited from task file's own example, CLI default is 25.
LANE COMPLETE. Working directly on main (no feature branch to merge per earlier decision).

STRETCH TASK (post-MVP, user-requested): dense-reward retrain with VecNormalize.
Commit 6a22cab "feat(sim_policy): switch to dense reward + VecNormalize for retrain" -- reviewed and approved (all 6 flagged normalization-correctness risks independently verified against SB3 source, no bugs found). Smoke-tested at 2000 timesteps / 2 episodes before real run.
Next: launch real 1,500,000-timestep training run as a detached OS process (not subagent-polled, learned from earlier token waste). Will verify completion via PPO.load(...).num_timesteps directly, not trust self-reports.

Real training run COMPLETE. num_timesteps=1507328 (verified via PPO.load()), ~3.7h wall-clock, ep_rew_mean 13.6->38.6 over the run (real dense-reward learning signal). ppo_v1.zip and vecnormalize.pkl both updated (17:31). rollout.json/demo.mp4 on disk are still stale (from the 2-episode smoke test / old MVP run) -- need full 25-episode evaluate.py + record_demo.py re-run next, then pytest, then commit, then README update with real (possibly still low) success_rate numbers -- do not round up.

STRETCH TASK COMPLETE. Full 25-episode evaluate.py + record_demo.py re-run against the real 1.5M-timestep model. Results (verified directly, not from a subagent report):
- success_rate: 0.0 (still 0/25 episodes) -- reported honestly, not rounded up
- mean_reward: 30.27 (up from -20 on the sparse MVP run)
- mean_episode_length: 200.0 (every episode still runs full duration, no early success termination)
- demo.mp4 verified valid (201 frames, 6.7s, 480x480)
- pytest sim_policy/tests/ -- 2/2 PASS
- README updated + committed: 52dfa81 "docs: update sim_policy README status with dense-reward retrain results"
No source-code changes needed beyond the already-reviewed 6a22cab commit (evaluate.py/record_demo.py just re-run against new artifacts, which are gitignored).
