import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from detections.dagmm_detection import DAGMMDetection
from dagmm.delta_dagmm.delta_dagmm import DELTA_DAGMM
from flwr.common import parameters_to_ndarrays

class DeltaDAGMMDetection(DAGMMDetection):
    def __init__(self, config):
        super().__init__(config)
        self.do_test_data_collection = config.get("do_test_data_collection")

    def _perform_data_collection(self, client_ids, client_updates, global_model):
        # Perform data collection needed for DeltaDAGMM training.
        
        global_model_arrays = parameters_to_ndarrays(global_model)
        last_layer_global_weights = global_model_arrays[-2]
        last_layer_global_biases = global_model_arrays[-1]

        self.iteration += 1

        if self.do_test_data_collection:
            # Do test data collection: Also save the global model
            flat_global = np.concatenate([last_layer_global_weights.flatten(), last_layer_global_biases.flatten()])
            np.save(os.path.join(self.dagmm_output_dir, f"update_iter{self.iteration}_global.npy"), flat_global)

        for cid, update in zip(client_ids, client_updates):
            last_layer_local_weights = update[-2]
            last_layer_local_biases = update[-1]

            # Calculate delta weights
            delta_weights = last_layer_local_weights - last_layer_global_weights
            delta_biases = last_layer_local_biases - last_layer_global_biases

            flat = np.concatenate([delta_weights.flatten(), delta_biases.flatten()])
            np.save(os.path.join(self.dagmm_output_dir, f"update_iter{self.iteration}_client{cid}.npy"), flat)

    def _perform_detection(self, client_ids, client_updates, global_model):
        # preprocess client updates and the global model
        # apply Delta-DAGMM to detect anomalies
        # return only non-anomalous updates

        global_model_arrays = parameters_to_ndarrays(global_model)
        last_layer_global_weights = global_model_arrays[-2]
        last_layer_global_biases = global_model_arrays[-1]
        
        # Flatten updates
        inputs = []
        for update in client_updates:
            last_layer_local_weights = update[-2]
            last_layer_local_biases = update[-1]

            # Calculate delta weights
            delta_weights = last_layer_local_weights - last_layer_global_weights
            delta_biases = last_layer_local_biases - last_layer_global_biases

            flat = np.concatenate([delta_weights.flatten(), delta_biases.flatten()])
            inputs.append(flat)
        inputs = np.stack(inputs)

        batch_size = len(inputs)
        if batch_size > 128:
            batch_size = 128

        data_loader = DataLoader(TensorDataset(torch.tensor(inputs, dtype=torch.float32)),
                              batch_size=batch_size, shuffle=False)

        # Get energies from Delta-DAGMM model (high energy -> anomaly)
        energies = self.model.compute_energies(data_loader, self.gmm_params)        
        
        print(f"Threshold: {self.threshold:.2f}")
        sorted_energies = [int(energy) for energy in sorted(energies)]
        print(f"Energies: ", sorted_energies)

        # Keep clients with low anomaly score
        kept_ids = [cid for cid, score in zip(client_ids, energies) if score < self.threshold]
        return kept_ids
    
    def _load_model(self):
        # Load trained Delta-DAGMM model
        print("Loading Delta-DAGMM model ...")
        model = DELTA_DAGMM(self.device, self.dagmm_hyperparameters)  # Load Delta-DAGMM model architecture
        model.load_state_dict(torch.load(self.dagmm_model_path, map_location=self.device))  # Load saved model parameters
        model.to(self.device)
        model.eval()

        self.gmm_params["mixture"] = torch.load(self.gmm_params_paths["mixture"])
        self.gmm_params["mean"] = torch.load(self.gmm_params_paths["mean"])
        self.gmm_params["cov"] = torch.load(self.gmm_params_paths["cov"])

        print("Loading completed!")
        return model