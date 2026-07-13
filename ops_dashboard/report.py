from datetime import datetime, timezone


def generate_report(rollout: dict) -> str:
    summary = rollout["summary"]
    lines = [
        f"# PoC Deployment Report — {rollout['task']}",
        "",
        f"**Policy:** `{rollout['policy_id']}`  ",
        f"**Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        f"- Success rate: **{summary['success_rate']:.1%}**",
        f"- Mean reward: **{summary['mean_reward']:.2f}**",
        f"- Mean episode length: **{summary['mean_episode_length']:.1f} steps**",
        f"- Episodes evaluated: **{summary['num_episodes']}**",
        "",
        "## Episode detail",
        "",
        "| Episode | Success | Reward | Length |",
        "|---|---|---|---|",
    ]
    for ep in rollout["episodes"]:
        mark = "yes" if ep["success"] else "no"
        lines.append(f"| {ep['episode_id']} | {mark} | {ep['total_reward']:.2f} | {ep['length']} |")
    lines += [
        "",
        "## Notes for the research team",
        "",
        "- This run used a simulated MuJoCo environment (AdroitHand); no physical hardware was involved.",
        "- `final_object_pose` for each episode comes from simulator ground truth, not a physical sensor.",
    ]
    return "\n".join(lines)
