from pathlib import Path

import streamlit as st

from data import DEFAULT_FIXTURE, load_rollout
from report import generate_report

REPO_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Mini-Dex FDE Ops Dashboard", layout="wide")
st.title("Mini-Dex — FDE Ops Dashboard")

REAL_ROLLOUT = REPO_ROOT / "sim_policy" / "artifacts" / "rollout.json"
rollout_path = REAL_ROLLOUT if REAL_ROLLOUT.exists() else DEFAULT_FIXTURE
source_note = " (fixture — real training not yet run)" if rollout_path == DEFAULT_FIXTURE else ""
st.caption(f"Data source: `{rollout_path.relative_to(REPO_ROOT)}`{source_note}")

rollout = load_rollout(rollout_path)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Success rate", f"{rollout['summary']['success_rate']:.1%}")
col2.metric("Mean reward", f"{rollout['summary']['mean_reward']:.2f}")
col3.metric("Mean episode length", f"{rollout['summary']['mean_episode_length']:.1f}")
col4.metric("Episodes", rollout["summary"]["num_episodes"])

video_path = REPO_ROOT / rollout["video_path"]
if video_path.exists():
    st.video(str(video_path))
else:
    st.info(f"Demo video not found yet at `{rollout['video_path']}` — run sim_policy's record_demo.py.")

st.subheader("Episodes")
st.table(rollout["episodes"])

if st.button("Generate PoC deployment report"):
    report_md = generate_report(rollout)
    st.download_button("Download report.md", report_md, file_name="poc_deployment_report.md")
    st.markdown(report_md)
