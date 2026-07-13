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
        # This installed gymnasium-robotics version's Adroit "Object" joint is a
        # 6-DOF OBJTx/Ty/Tz + OBJRx/Ry/Rz joint, not the classic 7-DOF quaternion
        # free joint — there is no 4-element quaternion in qpos to slice. Use the
        # body's global xpos/xquat instead, which is always a proper quaternion
        # regardless of the joint's own parameterization.
        obj_id = env.unwrapped.model.body("Object").id
        position = env.unwrapped.data.xpos[obj_id].tolist()
        orientation = env.unwrapped.data.xquat[obj_id].tolist()
        episodes.append({
            "episode_id": episode_id,
            "success": success,
            "total_reward": total_reward,
            "length": length,
            "final_object_pose": {
                "position": position,
                "orientation": orientation,
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
