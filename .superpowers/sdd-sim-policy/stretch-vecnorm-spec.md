# Stretch task: dense-reward retrain with VecNormalize

Goal: improve on the MVP's 0% success rate (AdroitHandRelocateSparse-v1, 200k
timesteps, mean_reward -20, every episode running the full 200 steps — a
known difficulty of sparse-reward RL from scratch, not a bug).

Already confirmed by the controller (do not re-derive):
- `info["success"]` is present on every `step()` call for BOTH
  `AdroitHandRelocate-v1` (dense) and `AdroitHandRelocateSparse-v1` (sparse),
  and is computed identically in both: `goal_achieved = goal_distance < 0.1`
  (only the reward shaping differs between the two task IDs, not the success
  criterion). So `evaluate.py`'s existing
  `success = bool(info.get("success", total_reward > 0))` line already reads
  the correct flag — no logic change needed there, only add normalization.

## File 1: `sim_policy/train.py` — full replacement

```python
import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

gym.register_envs(gymnasium_robotics)

# Dense-reward variant. The MVP trained on AdroitHandRelocateSparse-v1 (only
# rewards goal completion), which produced 0% success after 200k timesteps —
# a known difficulty of sparse-reward RL from scratch. Dense reward gives PPO
# a gradient every step (distance-to-object, lift bonus, distance-to-target).
TASK_ID = "AdroitHandRelocate-v1"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def train(total_timesteps: int, seed: int = 0) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec_env(TASK_ID, n_envs=4, seed=seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    model_path = ARTIFACTS_DIR / "ppo_v1.zip"
    model.save(model_path)
    vec_env.save(ARTIFACTS_DIR / "vecnormalize.pkl")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = train(args.timesteps, args.seed)
    print(f"Saved model to {path}")
```

## File 2: `sim_policy/evaluate.py` — full replacement

```python
import argparse
import json
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

gym.register_envs(gymnasium_robotics)

TASK_ID = "AdroitHandRelocate-v1"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def evaluate(model_path: Path, vecnormalize_path: Path, num_episodes: int, policy_id: str) -> dict:
    env = gym.make(TASK_ID)
    model = PPO.load(model_path)

    # The policy was trained on normalized observations (see train.py's
    # VecNormalize wrapper). Load the saved running stats onto a throwaway
    # DummyVecEnv just to reuse VecNormalize.normalize_obs()/.load() — the
    # actual episode rollout below still uses the raw (non-vec) `env` so the
    # existing per-episode seeding and terminated/truncated loop are
    # untouched. training=False freezes the stats; norm_reward=False so the
    # reward we log is the real environment reward, not the training signal.
    vecnorm = VecNormalize.load(str(vecnormalize_path), DummyVecEnv([lambda: gym.make(TASK_ID)]))
    vecnorm.training = False
    vecnorm.norm_reward = False

    episodes = []
    for episode_id in range(num_episodes):
        obs, info = env.reset(seed=episode_id)
        total_reward = 0.0
        length = 0
        terminated = truncated = False
        success = False
        while not (terminated or truncated):
            norm_obs = vecnorm.normalize_obs(obs)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            length += 1
            success = bool(info.get("success", success))
        obj_id = env.unwrapped.model.body("Object").id
        episodes.append({
            "episode_id": episode_id,
            "success": success,
            "total_reward": total_reward,
            "length": length,
            "final_object_pose": {
                "position": env.unwrapped.data.xpos[obj_id].tolist(),
                "orientation": env.unwrapped.data.xquat[obj_id].tolist(),
            },
        })
    env.close()

    success_rate = sum(e["success"] for e in episodes) / len(episodes)
    mean_reward = sum(e["total_reward"] for e in episodes) / len(episodes)
    mean_length = sum(e["length"] for e in episodes) / len(episodes)

    return {
        "task": TASK_ID,
        "policy_id": policy_id,
        "episodes": episodes,
        "summary": {
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "mean_episode_length": mean_length,
            "num_episodes": len(episodes),
        },
        "video_path": "sim_policy/artifacts/demo.mp4",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ARTIFACTS_DIR / "ppo_v1.zip")
    parser.add_argument("--vecnormalize", type=Path, default=ARTIFACTS_DIR / "vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--policy-id", default="ppo_v1")
    args = parser.parse_args()

    result = evaluate(args.model, args.vecnormalize, args.episodes, args.policy_id)
    out_path = ARTIFACTS_DIR / "rollout.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
```

Note the changed `evaluate()` signature (added `vecnormalize_path` positional
param) — this is intentional, update the call site in `__main__` to match
(already shown above).

## File 3: `sim_policy/record_demo.py` — full replacement

```python
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

gym.register_envs(gymnasium_robotics)

TASK_ID = "AdroitHandRelocate-v1"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def record(model_path: Path, vecnormalize_path: Path, out_path: Path, seed: int = 0, fps: int = 30) -> Path:
    env = gym.make(TASK_ID, render_mode="rgb_array")
    model = PPO.load(model_path)

    # Same normalization as evaluate.py — the policy expects normalized
    # observations, otherwise it acts on out-of-distribution input.
    vecnorm = VecNormalize.load(str(vecnormalize_path), DummyVecEnv([lambda: gym.make(TASK_ID)]))
    vecnorm.training = False
    vecnorm.norm_reward = False

    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    terminated = truncated = False
    while not (terminated or truncated):
        norm_obs = vecnorm.normalize_obs(obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
    env.close()

    imageio.mimsave(out_path, frames, fps=fps)
    return out_path


if __name__ == "__main__":
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = record(
        ARTIFACTS_DIR / "ppo_v1.zip",
        ARTIFACTS_DIR / "vecnormalize.pkl",
        ARTIFACTS_DIR / "demo.mp4",
    )
    print(f"Saved video to {path}")
```

## Your job (implementer)

1. Replace the three files above with the exact code given (this is a
   transcription task, not a design task — the VecNormalize integration
   details are load-bearing and already decided).
2. Smoke-test the full pipeline with a TINY timestep budget before touching
   the real long run:
   - `sim_policy/.venv/Scripts/python.exe sim_policy/train.py --timesteps 2000`
     — confirm it completes, prints `Saved model to ...`, and that BOTH
     `sim_policy/artifacts/ppo_v1.zip` AND `sim_policy/artifacts/vecnormalize.pkl`
     exist afterward.
   - `sim_policy/.venv/Scripts/python.exe sim_policy/evaluate.py --episodes 2`
     — confirm it completes without error and prints `Wrote .../rollout.json`.
     (2 episodes is enough to smoke-test the normalization wiring; the real
     eval run with the full 25 episodes happens later, after the real
     training run, not now.)
   - `sim_policy/.venv/Scripts/pytest.exe sim_policy/tests/test_rollout_schema.py -v`
     — confirm both tests still PASS (schema didn't change, only how the
     rollout is produced).
   - You do NOT need to smoke-test `record_demo.py` in this pass — it shares
     the same normalization pattern as `evaluate.py`, which you're already
     verifying works; running it too would just add render time for the same
     wiring. Skip it here.
3. Commit `sim_policy/train.py`, `sim_policy/evaluate.py`,
   `sim_policy/record_demo.py` together in ONE commit (they're a matched set
   — the normalization contract only makes sense as a unit). Conventional
   commit message, e.g. `feat(sim_policy): switch to dense reward + VecNormalize for retrain`.
   Do NOT commit `sim_policy/artifacts/` (already gitignored, but double
   check `git status --short` before committing, per the git-safety note
   below).
4. Do NOT kick off the real long (1-2M timestep) training run — the
   controller will do that separately as a detached background process, once
   this smoke-tested commit is in. Leave `sim_policy/artifacts/ppo_v1.zip`
   and `vecnormalize.pkl` as whatever the 2000-timestep smoke test produced;
   the controller will overwrite them with the real run's output later.

## Git safety

Multiple parallel Claude sessions may be committing to `main` in this same
working tree concurrently (this has caused a cross-lane commit collision
before). When you `git add`, add ONLY the exact 3 files named above — never
`git add -A`, `git add .`, or `git add sim_policy/`. Run `git status --short`
immediately before committing and confirm only those 3 files are staged;
`git restore --staged <path>` anything else first.
