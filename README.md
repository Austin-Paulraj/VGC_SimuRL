# 🛡️ VGC_SimuRL: Multi-Agent RL for Competitive Singles

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Ray](https://img.shields.io/badge/Ray_RLlib-2.30+-028CF0?style=for-the-badge&logo=ray&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28+-00CC99?style=for-the-badge&logo=openai&logoColor=white)

**A high-performance Reinforcement Learning pipeline using Ray RLlib's New API Stack to master competitive Pokémon turn-based strategy.**

[🔗 Battle Data Exploration](#-project-timeline--discovery) · [🛠️ Library Usage](#-library-usage-guide) · [💻 Getting Started](#-getting-started)

</div>

---

## 📺 Project Preview
> **Core Mechanism:** The agent processes a vectorized "embedding" of the current battle state. To handle the unique constraints of Pokémon (where moves can be disabled or Pokémon can faint), we implement a **Valid Action Masking** system over a 10-way discrete action space.

<p align="center">
  <img src="https://raw.githubusercontent.com/poke-env/poke-env/master/docs/img/logo.png" width="400px" alt="Poke-env Infrastructure">
</p>

---

## 📌 Executive Summary

### The Problem
Pokémon battles present a "Massive Discrete" action space. At any turn, an agent has 10 potential choices (4 moves + 6 switches), but many are illegal based on the current state (e.g., choice-locked moves or fainted teammates).
* **Technical Gap:** Standard RL policies often attempt illegal moves, leading to `AssertionError: invalid action` crashes in the simulator.

### The Solution
**VGC_SimuRL** utilizes a modular Reinforcement Learning architecture:
1.  **Action Masking:** The `ActorCriticModule` applies a binary mask to the output logits, ensuring the agent only "sees" and samples legal moves.
2.  **Vectorized Embeddings:** Converts complex `Battle` objects (HP, status, types, and stat boosts) into fixed-length tensors for neural network processing.
3.  **Self-Play Multi-Agent Training:** Uses Ray RLlib to train two policies simultaneously, forcing the agents to adapt to competitive switching strategies.

---

## 🚀 Project Timeline & Discovery
The architecture of this project was dictated by the iterative analysis found in `Battle_Data_Exploration.ipynb`:

* **Phase 1: State Analysis:** Used various timesteps ($t$) to inspect `available_moves` and `available_switches`. Discovered that fainted Pokémon remain in the team array but must be masked out of the action space.
* **Phase 2: Reward Shaping:** Moved from sparse win/loss signals to a dense system. Final implementation values Pokémon fainting (`fainted_value=2.0`) and HP differentials (`hp_value=1.0`) to guide the gradient descent.
* **Phase 3: Robustness:** Encountered simulation crashes during early training loops; solved by implementing the `valid_action_mask` that synchronizes the NN output with the `poke-env` simulator's requirements.

---

## 🛠️ Library Usage Guide

| Library | Role in VGC_SimuRL |
| :--- | :--- |
| **Ray RLlib** | **Orchestration:** Manages the Multi-Agent environment, PPO algorithm updates, and parallelizing battle workers across CPU cores. |
| **Poke-env** | **Simulation:** The bridge to the Pokémon Showdown server. It parses the protocol into Pythonic objects like `Battle` and `Pokemon`. |
| **PyTorch** | **Deep Learning:** Defines the `ActorCriticModule` (The "Brain") which processes the state embeddings into action probabilities. |
| **Gymnasium** | **Interface:** Defines the `Box` (observations) and `Discrete(10)` (actions) spaces that allow the agent to communicate with the environment. |
| **TensorDict** | **Efficiency:** Used to bundle observations and action masks into unified data structures for optimized batch training. |

---

## 📂 Repository Layout
```text
VGC_SimuRL/
├── src/
│   ├── envs/            # SmogonEnv with reward & embedding logic
│   ├── agents/          # ActorCriticModule with NN architecture
│   └── training/        # PPO Config and Multi-Agent mapping scripts
├── notebooks/
│   ├── Battle_Data_Exploration.ipynb  # Timeline of battle data analysis
│   ├── Singles_env_trial.ipynb       # Environment & Reward prototyping
│   └── first_train_loop_trial.ipynb  # Primary training loop implementation
├── data/                # example_battle_dict.pickle & Team configurations
├── Dockerfile           # Setup for local Showdown + Python environment
└── requirements.txt     # Ray, Torch, and Poke-env dependencies

