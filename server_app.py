"""pytorch-example: A Flower / PyTorch app."""

import torch
from detections.strategies.detection_before_aggregation import FedAvgWithDetectionsBeforeAggregation
from detections.strategies.detection_after_aggregation import FedAvgWithDetectionsAfterAggregation
from detections.strategies.rffl import RFFL
from task import (
    Net,
    apply_test_transforms,
    get_weights,
    set_weights,
    test,
)
from torch.utils.data import DataLoader

from datasets import load_dataset
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from detections.detection_names import DetectionNames
from detection_handler import DetectionHandler

import yaml
with open("./config/dataset.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)
DATASET = dataset_config.get("dataset", [])

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def on_fit_config(server_round: int):
    """Construct `config` that clients receive when running `fit()`"""
    # NOTE: DEACTIVATED!
    lr = 0.01
    # Enable a simple form of learning rate decay TODO change
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


# Define metric aggregation function
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = context.run_config["server-device"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Prepare dataset for central evaluation

    # This is the exact same dataset as the one donwloaded by the clients via
    # FlowerDatasets. However, we don't use FlowerDatasets for the server since
    # partitioning is not needed.
    # We make use of the "test" split only
    print("\n----- DATASET -----")
    print(f"\t{DATASET}")
    print("----- ------- -----\n")
    global_test_set = load_dataset(DATASET)["test"]

    # Print the used attack
    attack_config = None
    with open("./config/attack_method.yaml", "r") as f:
        attack_config = yaml.safe_load(f)
    with open("./config/malicious_clients.yaml", "r") as f:
        malicious_clients_config = yaml.safe_load(f)
    print("\n+++++ ATTACK(S) +++++")
    print(f"\t{attack_config}")
    print(".......................")
    print(f"\t{malicious_clients_config}")
    print("+++++ +++++++++ +++++\n")

    # Load detection method from file
    with open("./config/detection_method.yaml", "r") as f:
        detection_method_config = yaml.safe_load(f)
    detection_method = detection_method_config.get("detection_method", [])
    print("\n===== DETECTION =====")
    print(f"\t{detection_method}")
    print("===== ========= =====\n")
    detection_method = detection_method_config.get("detection_method", [])
    detection_handler = DetectionHandler(
        detection_method,
        config=detection_method_config,
    )

    testloader = DataLoader(
        global_test_set.with_transform(apply_test_transforms),
        batch_size=32,
    )

    # Define strategy
    if detection_method == DetectionNames.rffl_detection.value:
        alpha = detection_method_config.get("alpha")
        beta = detection_method_config.get("beta")
        gamma = detection_method_config.get("gamma")
        strategy = RFFL(
            run_config=context.run_config,
            use_wandb=context.run_config["use-wandb"],
            detection_handler=detection_handler,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            initial_parameters=parameters,
            #on_fit_config_fn=on_fit_config,
            evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    elif detection_method == DetectionNames.wef_detection.value:
        from detections.strategies.wef import WEF
        strategy = WEF(
            run_config=context.run_config,
            use_wandb=context.run_config["use-wandb"],
            detection_handler=detection_handler,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            initial_parameters=parameters,
            #on_fit_config_fn=on_fit_config,
            evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    elif detection_handler.after_aggregation:
        # Perform the detection after the aggregation step
        strategy = FedAvgWithDetectionsAfterAggregation(
            run_config=context.run_config,
            use_wandb=context.run_config["use-wandb"],
            detection_handler=detection_handler,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            initial_parameters=parameters,
            #on_fit_config_fn=on_fit_config,
            evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        # Perform the detection before the aggregation step
        strategy = FedAvgWithDetectionsBeforeAggregation(
            run_config=context.run_config,
            use_wandb=context.run_config["use-wandb"],
            detection_handler=detection_handler,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            initial_parameters=parameters,
            #on_fit_config_fn=on_fit_config,
            evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)