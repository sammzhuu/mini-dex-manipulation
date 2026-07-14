### Task 5: Demo video recording

**Files:**
- Create: `sim_policy/record_demo.py`

**Interfaces:**
- Consumes: `sim_policy/artifacts/ppo_v1.zip`.
- Produces: `sim_policy/artifacts/demo.mp4`, referenced by `rollout.json`'s `video_path` field and displayed by `ops_dashboard/` (Lane D).

- [ ] **[MVP] Step 1: Write `sim_policy/record_demo.py`**

```python
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import imageio
from stable_baselines3 import PPO

gym.register_envs(gymnasium_robotics)

TASK_ID = "AdroitHandRelocateSparse-v1"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def record(model_path: Path, out_path: Path, seed: int = 0, fps: int = 30) -> Path:
    env = gym.make(TASK_ID, render_mode="rgb_array")
    model = PPO.load(model_path)

    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
    env.close()

    imageio.mimsave(out_path, frames, fps=fps)
    return out_path


if __name__ == "__main__":
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = record(ARTIFACTS_DIR / "ppo_v1.zip", ARTIFACTS_DIR / "demo.mp4")
    print(f"Saved video to {path}")
```

- [ ] **[MVP] Step 2: Run it**

Run: `python sim_policy/record_demo.py`
Expected: `sim_policy/artifacts/demo.mp4` exists and plays a short clip of the hand attempting the task.

- [ ] **Step 3: Commit**

```bash
git add sim_policy/record_demo.py
git commit -m "feat(sim_policy): add demo video recording script"
```

---

## Lane MVP complete when

- [ ] `sim_policy/artifacts/rollout.json` exists and passes `pytest sim_policy/tests/`.
- [ ] `sim_policy/artifacts/demo.mp4` exists and is a few seconds of visible hand/object motion (doesn't need a high success rate).

## Stretch (post-MVP, do only after MVP is done and other lanes are unblocked)

- Longer training run / hyperparameter tuning to raise `success_rate` meaningfully above the MVP run.
- Train and export a second task (e.g. `AdroitHandHammerSparse-v1`) so the dashboard has more than one scenario to pick from.
- Convert `demo.mp4` to an optimized `.gif` for the root README (smaller file, autoplays on GitHub).
- Add a `sim_policy/tests/test_env_check.py` that runs `env_check.py`'s logic as an actual pytest instead of a standalone script.
