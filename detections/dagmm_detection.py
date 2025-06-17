import os
import torch
import numpy as np
from dagmm.dagmm import DAGMM
from detections.detection import Detection

class DAGMMDetection(Detection):
    def __init__(self, config):
        # initialize
        self.do_data_collection = config.get("do_data_collection", [])
        self.dagmm_output_dir = config.get("dagmm_output_dir", [])
        os.makedirs(self.dagmm_output_dir, exist_ok=True)
    
        self.threshold = config.get("dagmm_threshold")
        self.dagmm_model_path = config.get("dagmm_model_path")
        self.dimensions = config.get("dagmm_dimensions")
        self.latent_dim = config.get("latent_dim")
        self.estimation_hidden_size = config.get("estimation_hidden_size")
        self.n_gmm = config.get("n_gmm")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.iteration = 0
        self.ignore_up_to = config.get("dagmm_ignore_up_to")

        if not self.do_data_collection:
            # Only load the model if no data collection is done
            self.model = self._load_model()

    def detect(self, server_round, client_ids, client_updates):
        if self.do_data_collection:
            self._perform_data_collection(client_ids, client_updates)
            return client_ids
        else:
            if server_round <= self.ignore_up_to:
                # Do not perform a detection up to the i'th round
                return client_ids
            else:
                return self._perform_detection(client_ids, client_updates)

    def _perform_data_collection(self, client_ids, client_updates):
        # Perform data collection needed for DAGMM training.
        self.iteration += 1
        for cid, update in zip(client_ids, client_updates):
            flat = np.concatenate([layer.flatten() for layer in update])
            np.save(os.path.join(self.dagmm_output_dir, f"update_iter{self.iteration}_client{cid}.npy"), flat)

    def _perform_detection(self, client_ids, client_updates):
        # preprocess client updates
        # apply DAGMM to detect anomalies
        # return only non-anomalous updates
        
        # Flatten updates
        inputs = []
        for update in client_updates:
            flat = np.concatenate([layer.flatten() for layer in update])
            inputs.append(flat)
        inputs = np.stack(inputs)

        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)

        batch_size = len(inputs)
        if batch_size > 128:
            batch_size = 128

        # Get energies from DAGMM model (high energy -> anomaly)
        energies = []
        with torch.no_grad():
            for i in range(0, len(inputs_tensor), batch_size):
                batch = inputs_tensor[i:i+batch_size]
                _, _, z, _ = self.model(batch)
                # Use the corrected compute_energy method using the *model's* stored GMM params
                energy, _ = self.model.compute_energy(z, size_average=False)
                energies.append(energy.to("cpu"))
        
        energies_ = torch.cat(energies).numpy() # These are the energies
        print("Threshold: ", self.threshold)
        sorted_energies = [int(energy) for energy in sorted(energies_)]
        print(f"Energies: ", sorted_energies)

        # Keep clients with low anomaly score
        kept_ids = [cid for cid, score in zip(client_ids, energies_) if score < self.threshold]
        return kept_ids

    def _load_model(self):
        # Load trained DAGMM model
        print("Loading DAGMM model ...")
        model = DAGMM(self.device, self.dimensions, self.latent_dim, self.estimation_hidden_size, self.n_gmm)  # Load DAGMM model architecture
        model.load_state_dict(torch.load(self.dagmm_model_path, map_location=self.device))  # Load saved model parameters
        model.to(self.device)
        model.eval()
        print("Loading completed!")
        return model

