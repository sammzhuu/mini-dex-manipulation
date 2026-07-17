import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

gym.register_envs(gymnasium_robotics)

# Dense-reward variant. The MVP trained on AdroitHandRelocateSparse-v1 (only
# rewards goal completion), which produced 0% success after 200k timesteps —
# a known difficulty of sparse-reward RL from scratch. Dense reward gives PPO
# a gradient every step (distance-to-object, lift bonus, distance-to-target).
TASK_ID = "AdroitHandRelocate-v1"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# The first dense-reward run (1.5M steps, default 64x64 MlpPolicy, n_envs=4
# via the default DummyVecEnv) reached mean reward +30.3 but 0/25 success:
# per-episode diagnostics showed the policy lifts the ball but never places
# it within the 0.1m success radius (best episode 0.18m, mean best-distance
# 0.33m across episodes) — a capacity/exploration ceiling, not a reward-
# shaping problem. SubprocVecEnv gives real multi-process parallelism
# (DummyVecEnv steps envs sequentially in one process, so raising n_envs
# there doesn't buy wall-clock throughput). The larger net and gSDE
# (state-dependent exploration) target that capacity/exploration ceiling.
POLICY_KWARGS = dict(net_arch=[256, 256])


def train(total_timesteps: int, n_envs: int, seed: int = 0, warm_start: Optional[Path] = None) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec_env(TASK_ID, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    if warm_start is not None:
        # Carry over the demo-derived obs normalization stats from BC
        # pretraining (see bc_pretrain.py) so fine-tuning starts from the
        # same input distribution the warm-started policy was calibrated
        # to, instead of resetting to VecNormalize's mean=0/var=1 default.
        bc_vecnorm_path = Path(warm_start).parent / "bc_vecnormalize.pkl"
        bc_stats = VecNormalize.load(str(bc_vecnorm_path), DummyVecEnv([lambda: gym.make(TASK_ID)]))
        vec_env.obs_rms = bc_stats.obs_rms

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        policy_kwargs=POLICY_KWARGS,
        use_sde=True,
        sde_sample_freq=4,
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.9,
    )

    if warm_start is not None:
        bc_model = PPO.load(warm_start)
        model.policy.load_state_dict(bc_model.policy.state_dict())

    model.learn(total_timesteps=total_timesteps)
    model_path = ARTIFACTS_DIR / "ppo_v1.zip"
    model.save(model_path)
    vec_env.save(ARTIFACTS_DIR / "vecnormalize.pkl")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warm-start", type=Path, default=None, help="Path to a BC-pretrained model (see bc_pretrain.py)")
    args = parser.parse_args()
    path = train(args.timesteps, args.n_envs, args.seed, args.warm_start)
    print(f"Saved model to {path}")
