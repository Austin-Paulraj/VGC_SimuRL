# 🛡️ VGC_SimuRL: Reinforcement Learning for Smogon Singles

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28+-00CC99?style=for-the-badge&logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**An end-to-end Reinforcement Learning pipeline for training autonomous agents to master Competitive Pokémon under Smogon Singles rules.**

[🔗 Poke-env Docs](https://poke-env.readthedocs.io/) · [📊 Smogon Tier Info](https://www.smogon.com/tiers/) · [🎮 Pokémon Showdown](https://pokemonshowdown.com/)

</div>

---

## 📺 Project Preview
> **Agent Logic:** The agent interfaces with a local Pokémon Showdown server via `poke-env`, processing complex battle states into tensors to decide between 4 moves or switching to 5 bench teammates.

<p align="center">
  <img src="https://raw.githubusercontent.com/poke-env/poke-env/master/docs/img/logo.png" width="400px" alt="VGC_SimuRL Logo Context">
</p>

---

## 📌 Executive Summary

### The Problem
Competitive Pokémon is a turn-based strategy game with **hidden information** (sets/items), **stochasticity** (accuracy/crits), and an astronomical state space.
* **Context:** Navigating the "Smogon Singles" format requires long-term positioning and predicting "switches."
* **Current Gap:** Standard heuristic bots (e.g., Max Damage) fail to account for late-game win conditions or defensive pivoting.

### The Solution
VGC_SimuRL implements a Reinforcement Learning (RL) framework that:
* **Rewards Positioning:** Uses a custom reward function factoring in HP differentials and fainted Pokémon.
* **Format Adherence:** Operates strictly within Smogon rulesets (Sleep Clause, Species Clause, etc.).
* **Scalable Training:** Interfaces directly with the `SinglesEnv` Gymnasium wrapper for rapid self-play or baseline evaluation.

---

## 🚀 Key Features
* **Custom Reward Shaping:** Moves beyond simple win/loss signals to include HP-based intermediate rewards.
* **Feature Engineering:** Vectorizes Pokémon types, stats, and active field effects into a neural-network-ready observation space.
* **Showdown Integration:** Fully compatible with local Pokémon Showdown private servers for high-speed simulation.

---

## 🛠️ Tech Stack & Architecture

### Built With
* **Core:** Python, PyTorch (Deep Learning).
* **Simulation:** `poke-env` (The Python-Showdown bridge).
* **Environment:** Farama Gymnasium (Standard RL Interface).

### Data Flow / Architecture
```mermaid
graph LR
A[Showdown Server] --> B[poke-env API]
B --> C{Gymnasium Env}
C --> D[Neural Network Strategy]
D --> E[Action: Move/Switch]
E --> A
