### Task 4: Evaluation + rollout export

**Files:**
- Create: `sim_policy/evaluate.py`
- Create: `sim_policy/tests/test_rollout_schema.py`
- Create: `sim_policy/.gitignore`

**Interfaces:**
- Consumes: `sim_policy/artifacts/ppo_v1.zip` from Task 3.
- Produces: `sim_policy/artifacts/rollout.json` matching `shared/schema/rollout_schema.json` — consumed by `ops_dashboard/` (Lane D).

- [ ] **[MVP] Step 1: Write `sim_policy/.gitignore`**

```
artifacts/
```

- [ ] **[MVP] Step 2: Write the failing test first**

```python
# sim_policy/tests/test_rollout_schema.py
import json
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "shared" / "schema" / "rollout_schema.json"
FIXTURE_PATH = REPO_ROOT / "shared" / "fixtures" / "rollout.json"


def test_fixture_matches_schema():
    schema = json.loads(SCHEMA_PATH.read_text())
    fixture = json.loads(FIXTURE_PATH.read_text())
    jsonschema.validate(instance=fixture, schema=schema)


def test_generated_rollout_matches_schema():
    generated_path = REPO_ROOT / "sim_policy" / "artifacts" / "rollout.json"
    if not generated_path.exists():
        pytest.skip("run evaluate.py first to generate artifacts/rollout.json")
    schema = json.loads(SCHEMA_PATH.read_text())
    generated = json.loads(generated_path.read_text())
    jsonschema.validate(instance=generated, schema=schema)
```

- [ ] **Step 3: Run it to verify the first test passes and the second skips**

Run: `pytest sim_policy/tests/test_rollout_schema.py -v`
Expected: `test_fixture_matches_schema PASSED`, `test_generated_rollout_matches_schema SKIPPED`.

- [ ] **[MVP] Step 4: Write `sim_policy/evaluate.py`**

```python
import argparse
import json
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO

gym.register_envs(gymnasium_robotics)

TASK_ID = "AdroitHandRelocateSparse-v1"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def evaluate(model_path: Path, num_episodes: int, policy_id: str) -> dict:
    env = gym.make(TASK_ID)
    model = PPO.load(model_path)

    episodes = []
    for episode_id in range(num_episodes):
        obs, info = env.reset(seed=episode_id)
        total_reward = 0.0
        length = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            length += 1
        success = bool(info.get("success", total_reward > 0))
        qpos = env.unwrapped.data.qpos
        episodes.append({
            "episode_id": episode_id,
            "success": success,
            "total_reward": total_reward,
            "length": length,
            "final_object_pose": {
                "position": qpos[-7:-4].tolist(),
                "orientation": qpos[-4:].tolist(),
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
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--policy-id", default="ppo_v1")
    args = parser.parse_args()

    result = evaluate(args.model, args.episodes, args.policy_id)
    out_path = ARTIFACTS_DIR / "rollout.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
```

Use the object-pose slice you confirmed in Task 2; the code above assumes the common `qpos[-7:-4]` / `qpos[-4:]` convention.

- [ ] **[MVP] Step 5: Run it**

Run: `python sim_policy/evaluate.py`
Expected: prints `Wrote .../sim_policy/artifacts/rollout.json`.

- [ ] **[MVP] Step 6: Run the test again to verify it passes**

Run: `pytest sim_policy/tests/test_rollout_schema.py -v`
Expected: both tests `PASSED`.

- [ ] **Step 7: Commit**

```bash
git add sim_policy/evaluate.py sim_policy/tests/test_rollout_schema.py sim_policy/.gitignore
git commit -m "feat(sim_policy): add evaluation script and rollout schema test"
```

---

