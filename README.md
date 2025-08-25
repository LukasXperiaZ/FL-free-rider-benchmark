# FL-free-rider-benchmark

## Usage:
Prerequisits:
* install the necessary dependencies via e.g. poetry and source the virtual environment

# Run a single experiment:
* set configuration variables in `generate_config.ipynb` and execute the notebook
* run `flwr run . local-sim-gpu` to launch the simulation

# Run multiple benchmark runs one after another:
* run `./bash_scripts/run_benchmark_run.sh <dataset> <detection>`
* e.g. `./bash_scripts/run_benchmark_run.sh mnist delta_dagmm`