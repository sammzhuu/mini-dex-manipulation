import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import minari
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train import ARTIFACTS_DIR, POLICY_KWARGS, TASK_ID

gym.register_envs(gymnasium_robotics)

# From-scratch PPO (see train.py) plateaus at 0/25 success: it learns to
# lift the object but never learns the fine positional precision needed to
# land it within the 0.1m target radius, and more steps/tuning alone don't
# close that gap (verified: best-approach distance is statistically
# unchanged before/after a 10M-step retrain). D4RL/relocate/expert-v2 (via
# Minari, same AdroitHandRelocate-v1 env/obs/action spaces) is a dataset of
# 5000 expert trajectories that succeed 96.5% of the time — behavior-cloning
# on it gives the policy a head start that already knows what "close the
# gap" looks like, before PPO fine-tuning takes over.
DEMO_DATASET_ID = "D4RL/relocate/expert-v2"


def load_demo_transitions(num_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = minari.load_dataset(DEMO_DATASET_ID, download=True)
    obs_chunks = []
    action_chunks = []
    for i, episode in enumerate(dataset):
        if i >= num_episodes:
            break
        # observations has one more entry than actions (includes the final
        # post-episode obs with no corresponding action).
        obs_chunks.append(episode.observations[:-1])
        action_chunks.append(episode.actions)
    obs = np.concatenate(obs_chunks).astype(np.float32)
    actions = np.concatenate(action_chunks).astype(np.float32)
    return obs, actions


def pretrain(num_episodes: int, epochs: int, batch_size: int, seed: int = 0) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    obs, actions = load_demo_transitions(num_episodes)

    # Seed VecNormalize's running obs stats directly from the demo data
    # (rather than SB3's default mean=0/var=1) so BC training and the PPO
    # fine-tune that follows both normalize observations the same way from
    # the first step, instead of the fine-tune shocking a BC-trained policy
    # with a different input distribution.
    vec_env = DummyVecEnv([lambda: gym.make(TASK_ID)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    vec_env.obs_rms.mean = obs.mean(axis=0)
    vec_env.obs_rms.var = obs.var(axis=0)
    vec_env.obs_rms.count = obs.shape[0]

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        seed=seed,
        policy_kwargs=POLICY_KWARGS,
        use_sde=True,
        sde_sample_freq=4,
    )
    policy = model.policy

    norm_obs = vec_env.normalize_obs(obs)
    obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=policy.device)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=policy.device)
    num_samples = obs_t.shape[0]

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        for start in range(0, num_samples, batch_size):
            idx = perm[start : start + batch_size]
            batch_obs = obs_t[idx]
            batch_actions = actions_t[idx]

            # deterministic=True -> distribution mode (the mean action),
            # differentiable w.r.t. policy params for both the standard
            # Gaussian and gSDE distributions used here (see
            # DiagGaussianDistribution.mode / StateDependentNoiseDistribution.mode).
            predicted = policy._predict(batch_obs, deterministic=True)
            loss = torch.nn.functional.mse_loss(predicted, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_obs.shape[0]
        print(f"epoch {epoch}: mse={epoch_loss / num_samples:.5f}")

    model_path = ARTIFACTS_DIR / "bc_pretrained.zip"
    model.save(model_path)
    vec_env.save(ARTIFACTS_DIR / "bc_vecnormalize.pkl")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = pretrain(args.num_episodes, args.epochs, args.batch_size, args.seed)
    print(f"Saved BC-pretrained model to {path}")
