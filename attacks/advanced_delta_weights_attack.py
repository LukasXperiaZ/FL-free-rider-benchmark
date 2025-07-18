from task import set_weights
from attacks.attack_client import AttackClient
import torch
from flwr.common import ArrayRecord
import json

class AdvancedDeltaWeightsAttack(AttackClient):

    def get_tensor_parameters(self, parameters):
        """Converts a list of NumPy arrays to a list of PyTorch tensors."""
        return [torch.tensor(param).float().to(self.device).clone().detach() for param in parameters]

    def get_parameters(self):
        return [val.cpu().clone().detach() for _, val in self.net.state_dict().items()]


    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        key = "previous_params"
        
        # Advanced delta weights attack
        current_params_tensor = self.get_tensor_parameters(parameters)
        previous_params_tensor = self._load_param_tensors(key)

        # Gaussian noise parameters
        mu = 0
        sigma = 1e-3

        if not previous_params_tensor:
            ### This is the first round, only add gaussian noise

            # Generate Gaussian noise
            noise = [(torch.randn_like(param) + mu) * sigma for param in current_params_tensor]

            # Add the noise to the current parameters
            new_params_tensor = [param + n for param, n in zip(current_params_tensor, noise)]

        else:
            ### This is after the first round, previous parameters are available!

            # Compute the delta of the weigts of the previous 
            # --- Note that we changed this from (prev - current) to (current - prev), maybe there is a mistake in Equation 2 in the paper of Lin et al. ---
            delta_weights = [current_param - prev_param for current_param, prev_param in zip(current_params_tensor, previous_params_tensor)]

            # Generate Gaussian noise
            noise = [(torch.randn_like(dw) + mu) * sigma for dw in delta_weights]

            # Add the delta weights and noise to the current parameters to simulate the update
            new_params_tensor = [current_param_tensor + delta_weight + n for current_param_tensor, delta_weight, n in zip(current_params_tensor, delta_weights, noise)]

        new_params = [param.cpu().numpy() for param in new_params_tensor]

        # Set the current parameters as the previous for the next round
        self._save_params(key, parameters)

        label_counts_dict = self._get_label_distribution(self.trainloader)
        return self._fit_return(new_params, len(self.trainloader.dataset), 0.5, self.partition_id, json.dumps(label_counts_dict))
    
    def _save_params(self, key: str, parameters):
        arr_record = ArrayRecord.from_numpy_ndarrays(parameters)
        self.client_state[key] = arr_record

    def _load_param_tensors(self, key: str):
        if key not in self.client_state.keys():
            # No parameters stored
            return None

        parameters = self.client_state[key].to_numpy_ndarrays()

        return self.get_tensor_parameters(parameters)

