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

PROJECT_NAME = "FLOWER-experiment-attacks-detections"


class FedAvgWithDetections(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global  model when a new best is found, (3) logs
    results to W&B if enabled.
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.use_wandb = use_wandb
        # Initialise W&B if set
        if use_wandb:
            self._init_wandb_project()

        # Keep track of best acc
        self.best_acc_so_far = 0.0

        # A dictionary to store results as they come
        self.results = {}

        # Load detection method from file
        with open("./config/detection_method.yaml", "r") as f:
            detection_method_config = yaml.safe_load(f)
        detection_method = detection_method_config.get("detection_method", [])
        self.detection_handler = DetectionHandler(
            detection_method,
            config=detection_method_config,
        )
        # A set of banned clients IDs (permanently excluded)
        self.banned_client_ids = set()
        self.banned_partition_ids = set()


    def aggregate_fit(self, server_round, results, failures):
        
        client_updates = []
        client_ids = []
        flower_cid_to_partition_id = {}

        for client_proxy, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            cid = client_proxy.cid
            client_updates.append(weights)
            client_ids.append(cid)
            # Ensure 'partition_id' is in metrics, provide a fallback if not
            partition_id = fit_res.metrics.get("partition_id", "UNKNOWN_PARTITION_ID")
            flower_cid_to_partition_id[cid] = partition_id
  

        # Detect anomalies
        kept_client_ids = self.detection_handler.detect_anomalies(server_round, client_ids, client_updates)

        # Filter results
        filtered_results = [
            (client_proxy, fit_res)
            for client_proxy, fit_res in results
            if client_proxy.cid in kept_client_ids
        ]

        # Determine dropped clients
        dropped_ids = sorted(set(client_ids) - set(kept_client_ids))
        self.banned_client_ids.update(dropped_ids)

        dropped_partition_ids = set([flower_cid_to_partition_id[cid] for cid in sorted(dropped_ids)])
        self.banned_partition_ids.update(dropped_partition_ids)

        assert len(dropped_ids) == len(dropped_partition_ids)

        if dropped_partition_ids:
            print(f"[Round {server_round}] Dropped clients (anomalous): {dropped_partition_ids}")
        else:
            print(f"[Round {server_round}] No clients dropped.")

        print(f"All anomalous clients detected and removed: {sorted(list(self.banned_partition_ids))}")

        # Continue with filtered results
        return super().aggregate_fit(server_round, filtered_results, failures)
    
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
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, round, accuracy, parameters):
        """Determines if a new best global model has been found.

        If so, the model checkpoint is saved to disk.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead we are going to apply them to a PyTorch
            # model and save the state dict.
            # Converts flwr.common.Parameters to ndarrays
            ndarrays = parameters_to_ndarrays(parameters)
            model = Net()
            set_weights(model, ndarrays)
            # Save the PyTorch model
            file_name = f"model_state_acc_{accuracy}_round_{round}.pth"
            torch.save(model.state_dict(), self.save_path / file_name)

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