import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from dagmm.dagmm.dagmm import DAGMM
from detections.detection import Detection
import yaml

class DAGMMDetection(Detection):
    def __init__(self, config):
        # initialize
        self.do_data_collection = config.get("do_data_collection", [])
        if self.do_data_collection:
            self.dagmm_output_dir = config.get("dagmm_output_dir", [])
            os.makedirs(self.dagmm_output_dir, exist_ok=True)
            print("#######################################")
            print("Performing data collection ...")
            print("#######################################")
        
        self.iteration = 0

        if not self.do_data_collection:
            # Only perform these steps if no data collection is done
            with open(config.get("dagmm_threshold_path"), "r") as f:
                dagmm_threshold_config = yaml.safe_load(f) 
            self.threshold = dagmm_threshold_config.get("dagmm_threshold", [])
            self.dagmm_model_path = config.get("dagmm_model_path")

            hyperparameters_path = config.get("dagmm_hyperparameters_path")
            from dagmm.dagmm.dagmm import DAGMM_Hyperparameters
            self.dagmm_hyperparameters = DAGMM_Hyperparameters.load_params(hyperparameters_path)

            self.gmm_params_paths = config.get("gmm_parameters_paths")
            self.gmm_params = {}
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.ignore_up_to = config.get("dagmm_ignore_up_to")
        
            self.model = self._load_model()

    def detect(self, server_round, client_ids, client_updates, global_model):
        if self.do_data_collection:
            self._perform_data_collection(client_ids, client_updates, global_model)
            return client_ids
        else:
            if server_round <= self.ignore_up_to:
                # Do not perform a detection up to the i'th round
                return client_ids
            else:
                return self._perform_detection(client_ids, client_updates, global_model)

    def _perform_data_collection(self, client_ids, client_updates, global_model):
        # Perform data collection needed for DAGMM training.
        self.iteration += 1
        for cid, update in zip(client_ids, client_updates):
            flat = np.concatenate([layer.flatten() for layer in update])
            np.save(os.path.join(self.dagmm_output_dir, f"update_iter{self.iteration}_client{cid}.npy"), flat)

    def _perform_detection(self, client_ids, client_updates, global_model):
        # preprocess client updates
        # apply DAGMM to detect anomalies
        # return only non-anomalous updates
        
        # Flatten updates
        inputs = []
        for update in client_updates:
            flat = np.concatenate([layer.flatten() for layer in update])
            inputs.append(flat)
        inputs = np.stack(inputs)

        batch_size = len(inputs)
        if batch_size > 128:
            batch_size = 128

        data_loader = DataLoader(TensorDataset(torch.tensor(inputs, dtype=torch.float32)),
                              batch_size=batch_size, shuffle=False)

        # Get energies from DAGMM model (high energy -> anomaly)
        energies = self.model.compute_energies(data_loader, self.gmm_params)        
        
        print(f"Threshold: {self.threshold:.2f}")
        sorted_energies = [int(energy) for energy in sorted(energies)]
        print(f"Energies: ", sorted_energies)

        # Keep clients with low anomaly score
        kept_ids = [cid for cid, score in zip(client_ids, energies) if score < self.threshold]
        return kept_ids

    def _load_model(self):
        # Load trained DAGMM model
        print("Loading DAGMM model ...")
        model = DAGMM(self.device, self.dagmm_hyperparameters)  # Load DAGMM model architecture
        model.load_state_dict(torch.load(self.dagmm_model_path, map_location=self.device))  # Load saved model parameters
        model.to(self.device)
        model.eval()

        self.gmm_params["mixture"] = torch.load(self.gmm_params_paths["mixture"])
        self.gmm_params["mean"] = torch.load(self.gmm_params_paths["mean"])
        self.gmm_params["cov"] = torch.load(self.gmm_params_paths["cov"])

        print("Loading completed!")
        return model

