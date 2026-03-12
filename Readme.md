# Do You Really Think About Consequences?

**Bridging Reinforcement Learning and Control Theory for Long-Term Decision-Making in Beam Steering**

[![View Poster Online](https://img.shields.io/badge/📄_Poster-View_Online-6c8cff?style=for-the-badge)](https://mathphyssim.github.io/Do_you_really_think_about_consequences/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> Poster presented at [RL4AA'25 Workshop](https://rl4aa.github.io/RL4AA25/) at DESY, April 2025.
>
> **Authors:** O. Mironova, L. Fischl, T. Gallien, S. Hirlaender

## Abstract

This poster evaluates decision-making under delayed consequences using Reinforcement Learning (RL) and control methods in a linear accelerator (linac) beam steering setup. We demonstrate the strengths of various approaches in achieving optimal beam alignment while considering the constraints and dynamics of the system. Our key contribution lies in bridging RL and control theory to develop algorithms that effectively balance short-term rewards with long-term performance.

## Motivation

Beam steering in a linear accelerator serves as an illustrative example where classical greedy optimization methods may fall short due to their ignorance of long-term effects. Although the underlying dynamics are simulated as linear, the problem becomes non-trivial due to action constraints (corrector strength limits) and episode termination conditions (beam offset exceeding limits or reaching a target RMS threshold).

## Getting Started

```bash
# Install dependencies
pip install -e .

# Or using requirements.txt
pip install -r requirements.txt

# Train a PPO agent
python Run_PPO_training.py

# Run GP-MPC with structured model
python GP_MPC_approach_structured_generate-training_data.py

# Evaluate all policies
python Run_training_and_tests.py

# Analyse results
python Read_results_and_create_figures.py
```

## Code Structure

| Component | Files | Description |
|-----------|-------|-------------|
| **Environment** | `awake_steering_simulated.py` | Gymnasium-based simulation with linear dynamics, noise injection, and termination logic |
| **MPC Controllers** | `helper_scripts/MPC.py`, `linear_Bayesian_mpc.py`, `gp_mpc_*.py` | Perfect-model MPC, Bayesian linear MPC, standard and causal GP-MPC |
| **RL Agent** | `Run_PPO_training.py` | PPO training via stable-baselines3 |
| **Classical Optimizer** | `Run_stepwise_optimization.py` | Model-free stepwise optimization (COBYLA) |
| **Evaluation** | `Run_training_and_tests.py` | Multi-policy evaluation across noise levels |
| **Analysis** | `Read_results_and_create_figures.py` | Comparative performance plots |
| **Data Management** | `helper_scripts/data_management.py` | Shared `TrajectoryDataManager` and experiment folder utilities |
| **Config** | `config/*.yaml` | Environment and controller parameters |

## Control Policies Compared

1. **Analytic** — Direct matrix inversion of the known response matrix
2. **Model-Based MPC** — Finite-horizon optimization with the known linear model
3. **PPO** — Model-free RL trained to maximise cumulative reward
4. **Random** — Baseline using uniformly sampled actions
5. **Stepwise COBYLA** — Iterative model-free optimisation per step
6. **Linear Bayesian MPC** — Online linear model learning with Bayesian inference
7. **GP-MPC (Standard)** — Online non-parametric dynamics learning via GPs
8. **GP-MPC (Causal)** — Structured GP-MPC incorporating accelerator physics causality

## Citing This Work

```bibtex
@inproceedings{hirlaender2025consequences,
  title   = {Do You Really Think About Consequences? Bridging RL and Control Theory for Beam Steering},
  author  = {Mironova, O. and Fischl, L. and Gallien, T. and Hirlaender, S.},
  booktitle = {RL4AA'25 Workshop, DESY},
  year    = {2025}
}
```