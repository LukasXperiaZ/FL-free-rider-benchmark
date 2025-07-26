"""pytorch-example: A Flower / PyTorch app."""

import json
from logging import INFO

import torch
import wandb
from task import Net, create_run_dir, set_weights

from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg
from detection_handler import DetectionHandler
import yaml
import time

with open("run_name.yaml", "r") as f:
    run_name = yaml.safe_load(f).get("RUN_NAME")

with open("./config/malicious_clients.yaml", "r") as f:
    malicious_clients = yaml.safe_load(f).get("malicious_clients")

PROJECT_NAME = "FLOWER-experiment-attacks-detections"


class FedAvgWithDetections(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global  model when a new best is found, (3) logs
    results to W&B if enabled.

    Furthermore, it provides the basic functionality to perform detections.
    Concretely, one has to create a subclass of this class and implement the
    detection in aggregate_fit.
    When wanting to perform the aggregation, simply call super().aggregate_fit(...).
    To eliminate clients from the FL procedure, add them to self.banned_client_ids
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool, detection_handler: DetectionHandler, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config, run_name)
        self.use_wandb = use_wandb
        # Initialise W&B if set
        if use_wandb:
            self._init_wandb_project()

        # Keep track of best acc
        self.best_acc_so_far = 0.0

        # A dictionary to store results as they come
        self.results = {}

        self.detection_handler = detection_handler
        # A set of banned clients IDs (permanently excluded)
        self.banned_client_ids = set()
        self.banned_partition_ids = set()

        # Keeps track of the free riders that are detected in a round
        self.newly_detected_FR_partition_ids = []

        # Store the global model of the current round
        self.global_model = self.initial_parameters

        # Store the number of rounds
        self.num_rounds = run_config["num-server-rounds"]
        self.num_clients = None

        self.round_metrics = []

        self.start_time = time.time()


    def aggregate_fit(self, server_round, results, failures):
        if not self.num_clients:
            # This is the first round, store the number of participating clients
            self.num_clients = len(results)
        
        return super().aggregate_fit(server_round, results, failures)
    
    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.all()
        client_ids = list(clients.keys())

        # Filter out banned clients
        for client_id in client_ids:
            if client_id in self.banned_client_ids:
                client_manager.unregister(clients[client_id])

        clients = client_manager.all()
        if len(clients) == 0:
            raise RuntimeError("No eligible clients available (all banned).")

        return super().configure_fit(server_round, parameters, client_manager)

    def _init_wandb_project(self):
        # init W&B
        wandb.init(project=PROJECT_NAME, name=f"{run_name}_{str(self.run_dir)}")

    def _store_results(self, tag: str, results_dict):
        # Store relevant metrics of each round.
        round = results_dict["round"]
        accuracy = results_dict["centralized_accuracy"]
        detected_FR = self.newly_detected_FR_partition_ids

        round_metrics = RoundMetrics(round, accuracy, detected_FR)

        self.round_metrics.append(round_metrics)

        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

    def _update_best_acc(self, round, accuracy, parameters):
        """Determines if a new best global model has been found.

        If so, the model checkpoint is saved to disk.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            #logger.log(INFO, "💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead we are going to apply them to a PyTorch
            # model and save the state dict.
            # Converts flwr.common.Parameters to ndarrays
            #ndarrays = parameters_to_ndarrays(parameters)
            #model = Net()
            #set_weights(model, ndarrays)
            # Save the PyTorch model (not needed!)
            #file_name = f"model_state_acc_{accuracy}_round_{round}.pth"
            #torch.save(model.state_dict(), self.save_path / file_name)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

        if self.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=server_round)

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )

        if server_round == self.num_rounds:
            # After the last aggregation, store the final metrics
            self._store_final_metrics()

        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics
    
    def _store_final_metrics(self):
        """
        Store the final metrics needed for the evaluation.

        This method should be called after the last aggregation was performed.
        """
        # Record the elapsed time
        end_time = time.time()
        time_total = end_time - self.start_time
        time_per_iteration = time_total/self.num_rounds
        with open(f"{self.save_path}/time.json", "w", encoding="utf-8") as fp:
            json.dump({
                "time_per_iteration": time_per_iteration,
                "time_total": time_total
                }, fp)


        # Save the metrics of each round to disk.
        with open(f"{self.save_path}/round_metrics.json", "w", encoding="utf-8") as fp:
            json.dump([round_metric.get_dict() for round_metric in self.round_metrics], fp)      

        # Save the TP, FP and Precision to disk
        detected_FR = [id for id in self.banned_partition_ids if id in malicious_clients]
        TP = len(detected_FR)

        detected_BC = [id for id in self.banned_partition_ids if id not in malicious_clients]
        FP = len(detected_BC)

        undetected_FR = [id for id in malicious_clients if id not in self.banned_partition_ids]
        FN = len(undetected_FR)

        Precision = TP / (TP + FP) if TP != 0 else 0
        Recall = TP / (TP + FN) if TP != 0 else 0

        with open(f"{self.save_path}/Precision_Recall.json", "w", encoding="utf-8") as fp:
            json.dump({
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "Precision": Precision,
                "Recall": Recall
            }, fp)
    

from typing import List
class RoundMetrics():
    """
    A class for storing metrics of one round.
    """
    def __init__(self, round: int, accuracy: float, detected_FR: List):
        # The current federation round
        self.round = round
        # The accuracy of this round
        self.accuracy = accuracy
        # The newly(!) detected free riders of this round
        self.detected_FR = detected_FR

    def get_dict(self):
        return {
            "round": self.round,
            "accuracy": self.accuracy,
            "detected_FR": list(self.detected_FR)
        }