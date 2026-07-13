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
