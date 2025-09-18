# DSML: A Comprehensive Framework for Evaluating ML Methods for Distribution System State Estimation

The Robusttest repository provides the generation of distribtion system measurment data and the evaluation of the state estimation methods regarding their robustness against:
- Limited Measurement Data
- Missing Measurements
- Uncertain or Wrong Topologies
- Topological Changes

This project is based on the paper:

**“DSML: A Comprehensive Framework for Evaluating ML Methods for Smart Grid Condition Monitoring”**  
*T. Haug, C. Goebel, presented at ACM e-Energy 2025*

Read the full paper here:  
https://dl.acm.org/doi/10.1145/3679240.3734629

Or use this DOI link: https://doi.org/10.1145/3679240.3734629

Some basic functions to access the resulting pytorch.geometric data loaders are shown in 'example/example_NB.ipynb'.
Here it is also described how to use them to train a model, which if you have your own model also could be embedded here.
Furthermore, it is also described how to recreate the results shown in [DSML: A Comprehensive Framework for Evaluating ML Methods for Smart Grid Condition Monitoring](https://dl.acm.org/doi/10.1145/3679240.3734629).

The GAT model and loss functions used for model-driven learning were adapted from the [Deep Statistical Solver for Distribution System State Estimation](https://github.com/TU-Delft-AI-Energy-Lab/Deep-Statistical-Solver-for-Distribution-System-State-Estimation) repository developed by the TU Delft AI Energy Lab.

### Repo structure

```bash
.gitlab-ci.yml # CI/CD pipeline
pyproject.toml # packaging config
README.md
src/robusttest # forecasting python module
   |-- __init__.py
   |-- core # All implementation of core features
   |   |-- __init__.py
   |   |-- SE # State Estimation Methods
   |   |-- grid_time_series.py # Generation of measurement time series
   |-- interface # High Level Interface
   |   |-- __init__.py
   |   |-- ...
   |-- utils
   |   |-- __init__.py
example
   |-- example_NB.ipynb # examplary notebook
```

### Getting started

1. simply install with pip:

```bash 
pip install git+ssh://git@gitlab.lrz.de/energy-management-technologies-public/dsml.git@main
```

2. Tutorial for a basic forecast is in the notebook `example/testing.ipynb`


### For developers
When working on this project the following workflow is recommended.

1. Clone this repo.

```bash
pip install git+ssh://git@gitlab.lrz.de/energy-management-technologies-public/dsml.git
```

2. Inside the project folder create a virtual environent specific for this project. For example on linux:

```bash
cd dsml
python -m venv pyenv  # creates a virtual environment inside a directory called "pyenv"
source pyenv/bin/activate  # activates the environment
```

3. Install the `dsml` module in editable mode using `pip`. This makes use of the `pyproject.toml` and installs all dependencies. Thanks to the editable mode any changes you make in the source code of `robusttest` are applied immediately in the virtualenv.

```bash
pip install -e .
```

4. Create a branch, implement a fancy feature, issue a merge request, wait for it to be merged and have fun using the new feature !

Please comment your code extensively and comply with coding standards.