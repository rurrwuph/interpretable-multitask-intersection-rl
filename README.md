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
├── Network_scenario.py       # Inflow definitions, vehicle controllers, and Flow params
├── intersection_netw.py      # Parameterized 4-way intersection geometry (Scenarios A-I)
├── intersection_env.py       # Custom Gym environment with decomposed subtask rewards
├── multitask_dqn_model.py    # MultiTask DQN with slot encoder, shared trunk & subtask heads
├── replay_buffer.py          # Experience replay buffer with subtask reward vectors
├── train_multitask_dqn.py    # Training pipeline for multi-task DQN
├── train_ppo.py              # PPO baseline training script (Ray RLlib & Tune)
├── evaluate.py               # Evaluation harness across scenarios & rotating seeds
├── test.py                   # Model inference & policy sanity checks
├── results.py                # Metric aggregation, bar charts, and tabular reporting
├── correlate.py              # Subtask contribution and correlation analysis
├── diagnose_actuation.py     # Diagnostic tool for SUMO vehicle actuation
├── debug_dump_net.py         # Inspection utility for netconvert XML connections
├── smoke_test.py             # Rapid multi-scenario verification test
├── clean_ppo_curves.png      # PPO training convergence curve
├── fig3_learning_curves_comparison.png # Comparative Multi-Task DQN vs PPO curves
└── clean_ppo_metrics_5000.csv# Exported 5000-episode training metrics
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

# Generate Fig. 3, 4, 5 replication curves
python results.py
```

---

## Results & Reproduction
Evaluation metrics, learning curves, and comparative charts are reproduced in `fig3_learning_curves_comparison.png`, `clean_ppo_curves.png`, and `clean_ppo_metrics_5000.csv`.

<!-- validated benchmarks across scenarios A-I -->
