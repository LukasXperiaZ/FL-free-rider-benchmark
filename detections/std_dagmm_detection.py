import torch
from dagmm.std_dagmm.std_dagmm import STD_DAGMM
from detections.dagmm_detection import DAGMMDetection

class STDDAGMMDetection(DAGMMDetection):
    def __init__(self, config):
        super().__init__(config)

    def _load_model(self):
        # Load trained STD-DAGMM model
        print("Loading STD-DAGMM model ...")
        model = STD_DAGMM(self.device, self.dagmm_hyperparameters)  # Load DAGMM model architecture
        model.load_state_dict(torch.load(self.dagmm_model_path, map_location=self.device))  # Load saved model parameters
        model.to(self.device)
        model.eval()

        self.gmm_params["mixture"] = torch.load(self.gmm_params_paths["mixture"])
        self.gmm_params["mean"] = torch.load(self.gmm_params_paths["mean"])
        self.gmm_params["cov"] = torch.load(self.gmm_params_paths["cov"])

        print("Loading completed!")
        return model
