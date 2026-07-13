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
