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
# metrics_formatter_trial_0 = f'col_0'
# metrics_formatter_trial_1 = f'col_1'
# metrics_formatter_trial_2 = f'col_2'
# metrics_formatter_trial_3 = f'col_3'
# metrics_formatter_trial_4 = f'col_4'
# metrics_formatter_trial_5 = f'col_5'
# metrics_formatter_trial_6 = f'col_6'
# metrics_formatter_trial_7 = f'col_7'
# metrics_formatter_trial_8 = f'col_8'
# metrics_formatter_trial_9 = f'col_9'
# metrics_formatter_trial_10 = f'col_10'
# metrics_formatter_trial_11 = f'col_11'
# metrics_formatter_trial_12 = f'col_12'
# metrics_formatter_trial_13 = f'col_13'
# metrics_formatter_trial_14 = f'col_14'
# metrics_formatter_trial_15 = f'col_15'
# metrics_formatter_trial_16 = f'col_16'
# metrics_formatter_trial_17 = f'col_17'
# metrics_formatter_trial_18 = f'col_18'
# metrics_formatter_trial_19 = f'col_19'
# metrics_formatter_trial_20 = f'col_20'
# metrics_formatter_trial_21 = f'col_21'
# metrics_formatter_trial_22 = f'col_22'
# metrics_formatter_trial_23 = f'col_23'
# metrics_formatter_trial_24 = f'col_24'