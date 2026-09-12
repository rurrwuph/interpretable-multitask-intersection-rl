# Interpretable Multi-Task Intersection Reinforcement Learning

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Recreation and benchmark of the research paper:
> **"A Multi-Task Reinforcement Learning Approach for Navigating Unsignalized Intersections"**  
> *Kai et al., IEEE Intelligent Vehicles Symposium (IV) 2020.*

---

## Overview

Navigating unsignalized intersections presents significant challenges for autonomous vehicles due to complex vehicle interactions, occlusions, and diverse road geometries. This repository implements the paper's interpretable multi-task deep reinforcement learning framework, combining:
- **UC Berkeley Flow**: Traffic simulation abstraction layer.
- **Eclipse SUMO**: Microscopic traffic simulator for realistic vehicle dynamics and IDM controllers.
- **Multi-Task DQN**: Decomposed subtask reward modeling with shared representation learning and subtask-specific Q-value streams.
- **PPO Baseline (Ray RLlib & Tune)**: Baseline implementation using Ray RLlib with Ray Tune experiment management (`tune.run`), featuring `RLlibIntersectionWrapper`, multi-task environment cycling, and custom metric callbacks for tracking success rates and travel times.

---

## Visual Demonstration

### Autonomous Navigation in Scenario I (Extreme Saturation)
The recording below demonstrates the trained Multi-Task DQN navigating through the dense cross-traffic of **Scenario I** (North: 1000 veh/h, South: 1000 veh/h, East: 800 veh/h):

https://github.com/rurrwuph/interpretable-multitask-intersection-rl/raw/main/scenario_i_demo.mp4

<video src="scenario_i_demo.mp4" controls="controls" muted="muted" width="100%"></video>

> 🎬 **Watch / Download**: [`scenario_i_demo.mp4`](scenario_i_demo.mp4)  
> Recorded using [`record_demo.py`](record_demo.py) rendering SUMO simulation trajectories.

---

## Benchmark Scenarios

The framework dynamically generates the 9 unsignalized intersection topologies evaluated in Fig. 2 of the paper:

| Scenario | North Inflow (veh/h) | South Inflow (veh/h) | East Inflow (veh/h) | Lane Layout |
| :--- | :---: | :---: | :---: | :--- |
| **Scenario A** | 300 | 300 | 300 | Single lane per arm |
| **Scenario B** | 450 | 450 | 300 | Single lane, N-S dominant |
| **Scenario C** | 600 | 600 | 300 | Single lane, congested N-S |
| **Scenario D** | 300 | 600 | 600 | Multi-lane with turn pocket |
| **Scenario E** | 600 | 300 | 600 | Multi-lane with turn pocket |
| **Scenario F** | 600 | 600 | 600 | Multi-lane symmetric |
| **Scenario G** | 800 | 800 | 400 | High volume dual approach |
| **Scenario H** | 1000 | 600 | 600 | Asymmetric heavy flow |
| **Scenario I** | 1000 | 1000 | 800 | Extreme saturation |

---

## Codebase Architecture

```text
├── Network_scenario.py        # Inflow definitions, vehicle controllers, and Flow params
├── intersection_netw.py       # Parameterized 4-way intersection geometry (Scenarios A-I)
├── intersection_env.py        # Custom Gym environment with decomposed subtask rewards
├── multitask_dqn_model.py     # MultiTask DQN with slot encoder, shared trunk & subtask heads
├── replay_buffer.py           # Experience replay buffer with subtask reward vectors
├── train_multitask_dqn.py     # Training pipeline for multi-task DQN
├── train_ppo.py               # PPO baseline training script (Ray RLlib & Tune)
├── evaluate.py                # Evaluation harness across scenarios & rotating seeds
├── test.py                    # Model inference & policy sanity checks
├── results.py                 # Metric aggregation, bar charts, and tabular reporting
├── correlate.py               # Subtask contribution and correlation analysis
├── diagnose_actuation.py      # Diagnostic tool for SUMO vehicle actuation
├── debug_dump_net.py          # Inspection utility for netconvert XML connections
├── smoke_test.py              # Rapid multi-scenario verification test
├── record_demo.py             # Script to record video demonstrations of policy rollouts
├── plot_training_curves.py    # Extracts and plots comparative learning curves
├── scenario_i_demo.mp4        # Visual video demo of Scenario I navigation
├── fig_training_comparison.png# Training curve comparison across algorithms
├── clean_ppo_curves.png       # PPO training convergence curve
└── clean_ppo_metrics_5000.csv # Exported 5000-episode training metrics
```

---

## Installation

### 1. Prerequisites
- Python 3.7 or 3.8
- [Eclipse SUMO](https://eclipse.dev/sumo/) (v1.8.0 or higher recommended)
  ```bash
  export SUMO_HOME="/path/to/sumo"
  ```

### 2. Environment Setup
```bash
git clone https://github.com/rurrwuph/interpretable-multitask-intersection-rl.git
cd interpretable-multitask-intersection-rl
pip install -r requirements.txt
```

---

## Usage

### Training Multi-Task DQN
```bash
python train_multitask_dqn.py --scenario scenario_b --episodes 5000
```

### Training PPO Baseline (Ray RLlib & Tune)
```bash
# Trains PPO using Ray RLlib with Ray Tune experiment coordination
python train_ppo.py --episodes 5000
```

### Evaluation & Paper Benchmarks
```bash
# Evaluate across all 9 scenarios
python evaluate.py --checkpoint checkpoints/best_model.pt --episodes 1000 --rotating

# Generate replication curves and figures
python results.py
```

### Plotting Training Curves
```bash
python plot_training_curves.py
```

### Recording Visual Demonstrations
```bash
# Record rollout video in SUMO for a target scenario
python record_demo.py
```

---

## Results & Reproduction
Evaluation metrics, learning curves, and comparative charts are reproduced in:
- `fig_training_comparison.png`: Comparison of reward progression and stability.
- `clean_ppo_curves.png`: PPO baseline convergence curves.
- `clean_ppo_metrics_5000.csv`: Raw metric logs across 5,000 training episodes.
- `scenario_i_demo.mp4`: Rendered visual demonstration of the policy negotiating dense traffic in Scenario I.
