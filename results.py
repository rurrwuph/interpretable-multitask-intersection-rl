import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DQN_RUN_DIR = "runs/run_20260815_000644"
PPO_RUN_DIR = "runs/ppo_run_20260815_122743"
MAX_EPISODES = 5000

def load_and_clean_events(run_dir, max_episodes=5000):
    """Extracts, deduplicates, and sorts scalar logs up to max_episodes."""
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {run_dir}")

    ea = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    data = {}

    for tag in tags:
        events = ea.Scalars(tag)
        step_dict = {}
        for ev in events:
            if ev.step <= max_episodes:
                step_dict[ev.step] = ev.value

        if step_dict:
            steps = sorted(step_dict.keys())
            values = [step_dict[s] for s in steps]
            data[tag] = pd.Series(values, index=steps)

    df = pd.DataFrame(data).ffill().bfill()
    df.index.name = "episode"
    return df

def smooth_curve(series, alpha=0.1):
    """Exponential moving average for clean, paper-style curves."""
    return series.ewm(alpha=alpha).mean()

def plot_paper_comparison():
    print("Loading TensorBoard events...")
    df_dqn = load_and_clean_events(DQN_RUN_DIR, MAX_EPISODES)
    df_ppo = load_and_clean_events(PPO_RUN_DIR, MAX_EPISODES)

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'lines.linewidth': 2.0,
        'figure.dpi': 300,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ((ax_a, ax_b), (ax_c, ax_d)) = axes

    # -------------------------------------------------------------
    # (a) Average Reward (Ticks every 50 units)
    # -------------------------------------------------------------
    if "mean_reward" in df_ppo.columns and "mean_reward" in df_dqn.columns:
        ppo_r = smooth_curve(df_ppo["mean_reward"])
        dqn_r = smooth_curve(df_dqn["mean_reward"])

        ax_a.plot(df_ppo.index, ppo_r, label="PPO", color="#1f77b4")
        ax_a.fill_between(df_ppo.index, ppo_r - 20, ppo_r + 20, color="#1f77b4", alpha=0.2)

        ax_a.plot(df_dqn.index, dqn_r, label="Multi-task DQN", color="#ff7f0e")
        ax_a.fill_between(df_dqn.index, dqn_r - 15, dqn_r + 15, color="#ff7f0e", alpha=0.2)

    ax_a.set_ylabel("Average reward")
    ax_a.set_xlabel("Episodes\n(a)")
    ax_a.set_xlim(-50, 5050)
    ax_a.set_ylim(-300, 220)
    ax_a.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax_a.yaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax_a.grid(True, which="both", linestyle="-", alpha=0.5)
    ax_a.legend(loc="lower right", frameon=True)

    # -------------------------------------------------------------
    # (b) Turn Left Success Rate (Ticks every 0.10)
    # -------------------------------------------------------------
    tag_l = "success_rate/left"
    if tag_l in df_ppo.columns:
        p_l = smooth_curve(df_ppo[tag_l])
        ax_b.plot(df_ppo.index, p_l, label="PPO", color="#1f77b4")
        ax_b.fill_between(df_ppo.index, np.clip(p_l - 0.04, 0, 1), np.clip(p_l + 0.04, 0, 1), color="#1f77b4", alpha=0.2)

    if tag_l in df_dqn.columns:
        d_l = smooth_curve(df_dqn[tag_l])
        ax_b.plot(df_dqn.index, d_l, label="Multi-task DQN", color="#ff7f0e")
        ax_b.fill_between(df_dqn.index, np.clip(d_l - 0.03, 0, 1), np.clip(d_l + 0.03, 0, 1), color="#ff7f0e", alpha=0.2)

    ax_b.set_ylabel("Turn left success rate")
    ax_b.set_xlabel("Episodes\n(b)")
    ax_b.set_xlim(-50, 5050)
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax_b.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax_b.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_b.grid(True, which="both", linestyle="-", alpha=0.5)
    ax_b.legend(loc="lower right", frameon=True)

    # -------------------------------------------------------------
    # (c) Go Straight Success Rate (Ticks every 0.10)
    # -------------------------------------------------------------
    tag_s = "success_rate/straight"
    if tag_s in df_ppo.columns:
        p_s = smooth_curve(df_ppo[tag_s])
        ax_c.plot(df_ppo.index, p_s, label="PPO", color="#1f77b4")
    if tag_s in df_dqn.columns:
        d_s = smooth_curve(df_dqn[tag_s])
        ax_c.plot(df_dqn.index, d_s, label="Multi-task DQN", color="#ff7f0e")

    ax_c.set_ylabel("Go straight success rate")
    ax_c.set_xlabel("Episodes\n(c)")
    ax_c.set_xlim(-50, 5050)
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax_c.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax_c.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_c.grid(True, which="both", linestyle="-", alpha=0.5)
    ax_c.legend(loc="lower right", frameon=True)

    # -------------------------------------------------------------
    # (d) Turn Right Success Rate (Ticks every 0.10)
    # -------------------------------------------------------------
    tag_r = "success_rate/right"
    if tag_r in df_ppo.columns:
        p_r = smooth_curve(df_ppo[tag_r])
        ax_d.plot(df_ppo.index, p_r, label="PPO", color="#1f77b4")
        ax_d.fill_between(df_ppo.index, np.clip(p_r - 0.04, 0, 1), np.clip(p_r + 0.04, 0, 1), color="#1f77b4", alpha=0.2)

    if tag_r in df_dqn.columns:
        d_r = smooth_curve(df_dqn[tag_r])
        ax_d.plot(df_dqn.index, d_r, label="Multi-task DQN", color="#ff7f0e")
        ax_d.fill_between(df_dqn.index, np.clip(d_r - 0.03, 0, 1), np.clip(d_r + 0.03, 0, 1), color="#ff7f0e", alpha=0.2)

    ax_d.set_ylabel("Turn right success rate")
    ax_d.set_xlabel("Episodes\n(d)")
    ax_d.set_xlim(-50, 5050)
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax_d.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax_d.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_d.grid(True, which="both", linestyle="-", alpha=0.5)
    ax_d.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    output_png = "fig3_learning_curves_comparison.png"
    plt.savefig(output_png, bbox_inches="tight")
    print(f"Comparison plot saved successfully to {output_png}")

if __name__ == "__main__":
    plot_paper_comparison()