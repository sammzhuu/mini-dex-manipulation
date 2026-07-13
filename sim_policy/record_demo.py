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
