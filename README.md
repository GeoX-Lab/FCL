# FCL:A Domain-Agnostic Continual Learning Framework via Frequency Completeness Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.7-red)](https://pytorch.org/)

A PyTorch-based framework for implementing and evaluating domain-Agnostic continual learning, featuring multiple state-of-the-art methods with reproducible configurations. 

## About This Project
This code repository is built upon and inspired by the [PyCIL (PyTorch Continual Learning)](https://github.com/G-U-N/PyCIL) framework. We've adapted and extended their implementation to suit our specific research needs.
For the original implementation and full documentation, please visit the official PyCIL repository.


## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Supported Methods](#supported-methods)
- [File Structure](#file-structure)
- [Extending the Framework](#extending-the-framework)


## Features

✔ **Multiple Algorithms**: Implementations of 6+ continual learning methods  
✔ **Flexible Experimentation**: JSON-based configuration system  
✔ **Reproducible Results**: Multi-seed support with deterministic training  
✔ **Comprehensive Metrics**: Tracks accuracy, forgetting, and task performance  
✔ **Modular Design**: Easy to add new models or datasets  

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/continual-learning-framework.git
   cd continual-learning-framework
2. **Create a conda environment (recommended)**:
   ```bash
   conda create -n cl python=3.8
   conda activate cl
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

## Quick Start
#### Single Experiment
    python main.py --config ./exps/lwf.json
#### Batch Execution
    python run_methods.py
#### Custom Data Directory
    python main.py --config ./exps/ewc.json --data_dir_folder ./custom_data/
    
## Configuration
Configuration files are stored in ./exps/ with the following structure:
   
    {
      "prefix": "experiment",
      "dataset": "cifar100",
      "memory_size": 2000,
      "memory_per_class": 20,
      "fixed_memory": false,
      "shuffle": true,
      "init_cls": 10,
      "increment": 10,
      "model_name": "icarl",
      "convnet_type": "resnet32",
      "device": [0],
      "seed": [1993, 2022, 42]
    }
  

## Key Parameters

- `model_name`: Algorithm to use (e.g., "lwf", "ewc")
- `dataset`: Supported datasets (cifar100 by default)
- `init_cls`: Number of initial classes
- `increment`: Classes added per task
- `seed`: Random seeds for reproducibility

## Supported Methods

| Method    | Paper | Config File |
|-----------|-------|-------------|
| Finetune  | - | `finetune.json` |
| EWC | [Kirkpatrick et al. (2017)](https://arxiv.org/abs/1612.00796) | `ewc.json` |
| LwF | [Li & Hoiem (2017)](https://arxiv.org/abs/1606.09282) | `lwf.json` |
| BiC | [Wu et al. (2019)](https://arxiv.org/abs/1905.13260) | `bic.json` |
| WA | [Zhao et al. (2020)](https://arxiv.org/abs/2001.01578) | `wa.json` |
| COIL | [Liu et al. (2021)](https://arxiv.org/abs/2103.10339) | `coil.json` |

## File Structure
    .
    ├── exps/                   # Experiment configurations
    │   ├── finetune.json       # Baseline fine-tuning
    │   ├── ewc.json            # Elastic Weight Consolidation
    │   └── ...                 # Other method configs
    │
    ├── logs/                   # Auto-generated training logs
    │   └── [model]/[dataset]/  # Organized by experiment
    │
    ├── utils/                  # Core utilities
    │   ├── data.py             # Dataset implementations
    │   ├── data_manager.py     # Task scheduler
    │   ├── factory.py          # Model factory
    │   └── toolkit.py          # Evaluation metrics
    │
    ├── main.py                 # Single-experiment entry
    ├── run_methods.py          # Batch experiment runner
    ├── trainer.py              # Training pipeline
    └── README.md
    
## Extending the Framework
#### Adding New Methods
   1. Implement your model in `utils/factory.py`
   2. Create a config file in `exps/`
   3. Add to the supported methods list in `run_methods.py`
#### Adding New Datasets
   1. Implement dataset class in utils/data.py
   2. Update DataManager in utils/data_manager.py
   3. Test with:
   
     python main.py --config ./exps/custom.json --data_dir_folder ./new_dataset/
     


