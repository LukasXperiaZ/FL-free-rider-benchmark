from task import set_weights
from client_app import BenignClient
import torch

class AdvancedDeltaWeightsAttack(BenignClient):
    def __init__(self, net, client_state, trainloader, valloader, local_epochs, partition_id):
        super().__init__(net, client_state, trainloader, valloader, local_epochs, partition_id)
        self.previous_params_tensor = None

    def get_tensor_parameters(self, parameters):
        """Converts a list of NumPy arrays to a list of PyTorch tensors."""
        return [torch.tensor(param).float().to(self.device).clone().detach() for param in parameters]

    def get_parameters(self):
        return [val.cpu().clone().detach() for _, val in self.net.state_dict().items()]


    def fit(self, parameters, config):
        #print(f"[Client {self.partition_id}] fit, config: {config}")
        set_weights(self.net, parameters)
        
        # Delta Weights attack
        current_params_tensor = self.get_tensor_parameters(parameters)
        previous_params_tensor = self.previous_params_tensor

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
        self.previous_params_tensor = current_params_tensor

        return (new_params, 
                len(self.trainloader),
                {"partition_id": self.partition_id}
            )
