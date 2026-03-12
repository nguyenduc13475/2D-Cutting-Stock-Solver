# ✂️ 2D Cutting Stock Solver - Pro Edition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PuLP](https://img.shields.io/badge/Optimization-PuLP-yellow.svg)](https://coin-or.github.io/pulp/)

A complete software system that solves the **2D Cutting Stock Problem (2D CSP)**. This project combines Combinatorial Optimization, Heuristics, and Deep Reinforcement Learning, all packaged in a Clean OOP architecture with a user-friendly UI.

---

## 🌟 Approaches (Implemented Solvers)

The system integrates four problem-solving strategies, from Exact Math to AI:

1. **Simplex ILP (Integer linear programming):** 
    - Use the `PuLP` library to create accurate mathematical models. 
    - Auto-routing is performed across three sub-models (Model 1, Model 2, Model 3) depending on whether the input is single or multi-stock and the number of stock (finite or infinite).
2. **Column Generation:** 
    - Advanced Operations Research (OR) solutions address problems with large solution spaces.
    - This method combines solving the Master Problem and generating strips/patterns through Dual Variables (Shadow prices) calculations combined with heuristics.
3. **Greedy + Simulated Annealing (Heuristic + Meta-heuristic):**
    - The Bottom-Left Fill algorithm is combined with Simulated Metallurgy (SA) to optimize packing density and minimize excess area.
4. **Deep Reinforcement Learning (DRL):**
    - Use a pre-trained Neural Network (built using `PyTorch`) to predict pick-and-drop actions (select workpiece, select product, rotate).
    - It has an **Auto-Fallback** mechanism: It automatically reverts to the Greedy algorithm if the problem size exceeds the network's trained state space.

---

## 📂 System Architecture

The project has been refactored according to modern software engineering principles, completely separating core logic, solvers, and UI:

```text
📦 src
 ┣ 📂 core/         # Contains Data Classes/Schemas (Stock, Product, Solution)
 ┣ 📂 solvers/      # Algorithm implementation (Inherited from BaseSolver)
 ┃ ┣ 📜 base_solver.py
 ┃ ┣ 📜 column_generation.py
 ┃ ┣ 📜 greedy_sa.py
 ┃ ┣ 📜 lp_models.py
 ┃ ┗ 📜 rl_solver.py
 ┣ 📂 ui/           # Tkinter user interface
 ┗ 📂 utils/        # Geometry, graphical display, helper processing
📜 main.py          # Application Entry Point
📜 model.ckpt       # (Download required) Pre-trained Weights cho RL Solver

```

---

## 🚀 Installation & Usage

### 1. Environment settings

We recommend using a virtual environment:

```bash
git clone https://github.com/nguyenduc13475/2D-Cutting-Stock-Solver.git
cd 2D-Cutting-Stock-Solver
python -m venv venv
source venv/bin/activate  # (On Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 2. Download Pre-trained Model (for RL Solver)

Reinforcement Learning algorithms require a weight file.
👉 **[Download the `model.ckpt` file from the Releases section.](https://www.google.com/search?q=https://github.com/nguyenduc13475/2D-Cutting-Stock-Solver/releases)** and place it directly in the project's root directory (same level as `main.py`).

### 3. Start the application

```bash
python main.py
```

---

## 💡 Instructions on using the Import feature

You can import data in bulk for Stocks and Products via a `.txt` file. The structure of each line in the `.txt` file is as follows:
`[Width], [Height], [Quantity/Demand]`

*Example:*

```text
100, 200, 5
50, 50, 20
```

---

## 📄 Project Report

For a deeper look at the mathematical formulas (ILP, Column Generation) and the design architecture of the RL model, please refer to the course report:
👉 **[Read the detailed report (in Vietnamese)](https://www.google.com/search?q=docs/Cutting_Stock_2D_Report_VN.pdf)**

---

*Author: Nguyễn Văn Đức*