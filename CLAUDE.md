# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains DSML (Distribution System Machine Learning), a comprehensive framework for evaluating ML methods for distribution system state estimation. The project focuses on testing robustness against limited measurement data, missing measurements, uncertain topologies, and topological changes.

## Architecture

The main codebase is located in `dsml-main/` and follows this structure:

```
dsml-main/
├── src/robusttest/           # Main Python package
│   ├── core/                 # Core implementation
│   │   ├── SE/              # State Estimation methods (GAT, MLP, GNN, etc.)
│   │   ├── grid_utils/      # Grid utilities (uncertainty, topology perturbation)
│   │   ├── grid_time_series.py
│   │   └── semethod.py      # Abstract base class for SE methods
│   └── interface/           # High-level interface modules
│       ├── SE_grid_TS.py    # Main interface for SE with grid time series
│       ├── evaluate_SE.py   # Evaluation utilities
│       ├── trainer.py       # Training utilities
│       └── generate_*.py    # Data generation utilities
├── examples/                # Example notebooks and scripts
│   ├── example_NB.ipynb     # Main tutorial notebook
│   └── testing.ipynb       # Additional examples
└── pyproject.toml          # Project configuration and dependencies
```

### Key Components

- **State Estimation Methods** (`src/robusttest/core/SE/`): Multiple implementations including GAT-DSSE, MLP-DSSE, GNN-DSSE, and ensemble methods
- **Grid Utilities** (`src/robusttest/core/grid_utils/`): Tools for grid uncertainty modeling and topology perturbation
- **Interface Layer** (`src/robusttest/interface/`): High-level APIs for data generation, training, and evaluation
- **Abstract Base Classes**: `SEMethod` class in `semethod.py` provides common interface for all state estimation methods

## Development Commands

### Installation and Setup
```bash
# Install in editable mode (for development)
cd dsml-main
pip install -e .

# Install from git (for usage)
pip install git+ssh://git@gitlab.lrz.de/energy-management-technologies-public/dsml.git@main
```

### Code Quality and Testing
```bash
# Code formatting (Black)
black -l 100 ./src/robusttest/interface/
black -l 100 ./src/robusttest/core/

# Linting (Flake8)
flake8 --max-line-length=100 --ignore=E501,E712 ./src/robusttest/interface/
flake8 --max-line-length=100 --ignore=E501,E712 ./src/robusttest/core/
```

### Dependencies

Key dependencies include:
- **Deep Learning**: PyTorch, PyTorch Geometric, PyTorch Lightning
- **Scientific Computing**: NumPy, Pandas, SciPy, Scikit-learn
- **Power Systems**: PandaPower, PVLib
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Notebook Support**: IPython, Jupyter

## Getting Started

1. Start with the tutorial in `examples/example_NB.ipynb` to understand basic usage
2. Use `examples/testing.ipynb` for additional examples
3. The main entry point for state estimation tasks is through `SE_grid_TS.py`
4. All state estimation methods inherit from the `SEMethod` abstract base class

## Code Standards

- Line length: 100 characters (enforced by Black and Flake8)
- Code formatting: Black with line length 100
- Linting: Flake8 with specific ignores for E501 and E712
- Python version support: 3.8+