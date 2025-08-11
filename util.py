import random

def store_necessary_settings(NUM_CLIENTS, PERCENT_MALICIOUS, DATASET, ATTACK_METHOD, ADDITIONAL_ATTACK_CONFIG, DETECTION_METHOD, ADDITIONAL_DETECTION_CONFIG, NUM_ROUNDS, FRACTION_FIT, USE_WANDB):
    num_malicious = max(0, int(NUM_CLIENTS * PERCENT_MALICIOUS))
    malicious_clients = sorted(random.sample(range(NUM_CLIENTS), num_malicious))

    print(f"Selected {num_malicious} malicious clients out of {NUM_CLIENTS}")
    print("Malicious client IDs:", malicious_clients)

    import yaml

    # Store the dataset to use as a .yaml file
    with open("./config/dataset.yaml", "w") as f:
        yaml.dump({"dataset": DATASET.value}, f)

    # Store the list of malicious clients as a .yaml file
    with open("./config/malicious_clients.yaml", "w") as f:
        yaml.dump({"malicious_clients": malicious_clients}, f)

    # Store the attack method as a .yaml file
    with open("./config/attack_method.yaml", "w") as f:
        yaml.dump({"attack_method": ATTACK_METHOD.value}, f)
        if ADDITIONAL_ATTACK_CONFIG:
            yaml.dump(ADDITIONAL_ATTACK_CONFIG, f)

    # Store the detection method as a .yaml file
    with open("./config/detection_method.yaml", "w") as f:
        yaml.dump({"detection_method": DETECTION_METHOD.value}, f)
        if ADDITIONAL_DETECTION_CONFIG:
            yaml.dump(ADDITIONAL_DETECTION_CONFIG, f)

    import toml

    NUM_CORES = 24  # For 13th Gen Intel(R) Core(TM) i9-13900KF
    NUM_CPUS = 1

    # When running on GPU, assign an entire GPU for each client
    NUM_GPUS = 1/NUM_CLIENTS
    min_gpu_perc = 1/NUM_CORES

    if NUM_GPUS < min_gpu_perc:
        NUM_GPUS = min_gpu_perc
        
    print("num_gpus: ", NUM_GPUS)

    # Generate the name of the run.
    from evaluation.utils import get_output_path
    RUN_NAME = get_output_path(DATASET, NUM_CLIENTS, int(PERCENT_MALICIOUS*100), ATTACK_METHOD, DETECTION_METHOD)
    with open("run_name.yaml", "w") as f:
        yaml.dump({"RUN_NAME": RUN_NAME}, f)

    # Load pyproject.toml
    with open("pyproject.toml", "r") as f:
        data = toml.load(f)

    # Modify the values
    data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = NUM_ROUNDS
    data["tool"]["flwr"]["app"]["config"]["fraction-fit"] = FRACTION_FIT
    data["tool"]["flwr"]["app"]["config"]["use-wandb"] = USE_WANDB

    data["tool"]["flwr"]["federations"]["local-sim"]["options"]["num-supernodes"] = NUM_CLIENTS
    data["tool"]["flwr"]["federations"]["local-sim-gpu"]["options"]["num-supernodes"] = NUM_CLIENTS
    data["tool"]["flwr"]["federations"]["local-sim-gpu"]["options"]["backend"]["client-resources"]["num-cpus"] = NUM_CPUS
    data["tool"]["flwr"]["federations"]["local-sim-gpu"]["options"]["backend"]["client-resources"]["num-gpus"] = NUM_GPUS

    # Save changes back to pyproject.toml
    with open("pyproject.toml", "w") as f:
        toml.dump(data, f)
