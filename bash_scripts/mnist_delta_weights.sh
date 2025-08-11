#!/bin/bash
# This script runs all experiments for the delta weights detection with the mnist dataset.

# 0. Source the virtual environment
source .venv/bin/activate

# 1. generate the configuration
python generate_config.py 

# 2. run one experiment
flwr run . local-sim-gpu