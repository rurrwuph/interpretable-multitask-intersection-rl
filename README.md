# Interpretable Multi-Task Intersection RL

Recreation of the paper:
**"A Multi-Task Reinforcement Learning Approach for Navigating Unsignalized Intersections"** (Kai et al., IEEE Intelligent Vehicles Symposium (IV) 2020).

## Overview
This repository implements an interpretable multi-task deep reinforcement learning framework for autonomous vehicle navigation through unsignalized 4-way intersections with heterogeneous traffic.

Built using modified UC Berkeley Flow and Eclipse SUMO.

## Scenarios (Kai et al. Fig. 2)
The network models 9 distinct unsignalized intersection geometries:
- **Scenario A**: 1 through lane per arm, 300 veh/hr/arm
- **Scenario B**: 1 through lane, heavier North/South flow (450 veh/hr)
- **Scenario C**: 1 through lane, asymmetric high flow (600/600/300 veh/hr)
- **Scenario D - F**: Variable lane configurations with dedicated left turn pockets
- **Scenario G - I**: High saturation scenarios up to 1000 veh/hr

## Core Architecture
- **Multi-Task DQN**: Shared latent trunk with decomposed subtask value and advantage streams.
- **Safety Subtasks**: Time-to-Collision (TTC) penalty, velocity tracking, collision avoidance.
- **Baseline**: PPO baseline using Ray RLlib with Tune.

Benchmark evaluation harness under active development.
