import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

TASK_ID = "AdroitHandRelocateSparse-v1"


def main():
    env = gym.make(TASK_ID)
    obs, info = env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("env OK, obs shape:", obs.shape, "action shape:", action.shape)
    env.close()


if __name__ == "__main__":
    main()
