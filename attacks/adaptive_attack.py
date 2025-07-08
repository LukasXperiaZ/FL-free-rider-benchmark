from task import set_weights
from attacks.advanced_delta_weights_attack import AdvancedDeltaWeightsAttack
import torch
import numpy as np
from flwr.common import ArrayRecord
import random

class AdaptiveAttack(AdvancedDeltaWeightsAttack):
    """
    The adaptive attack is based on the advanced delta weights attack.
    The main difference is that it simulates normal training by adding different kinds of noise (with different standard deviations)
    to different parameters.
    Sadly, the authors do not specify how exactly that is done.
    """

    def _generate_adaptive_noise(self, like_tensor):
        # Function to generate adaptive noise, i.e. different kinds of noise for different parameters.

        # Gaussian noise parameters
        mu = 0
        sigmas = [1e-3, 1e-4, 1e-5]

        noise = []
        for tensor in like_tensor:
            random_mask = torch.rand(tensor.shape, device=tensor.device)
            sigma_0_mask = random_mask <= 1.0/3.0
            sigma_1_mask = (1.0/3.0 < random_mask) & (random_mask <= 2.0/3.0)
            sigma_2_mask = 2.0/3.0 < random_mask

            gaussian_0 = (torch.randn_like(tensor) + mu) * sigmas[0]
            gaussian_1 = (torch.randn_like(tensor) + mu) * sigmas[1]
            gaussian_2 = (torch.randn_like(tensor) + mu) * sigmas[2]

            noise_tensor = torch.zeros_like(tensor)
            noise_tensor[sigma_0_mask] += gaussian_0[sigma_0_mask]
            noise_tensor[sigma_1_mask] += gaussian_1[sigma_1_mask]
            noise_tensor[sigma_2_mask] += gaussian_2[sigma_2_mask]

            noise.append(noise_tensor)

        return noise


    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        key = "previous_params"
        
        # Adaptive attack
        current_params_tensor = self.get_tensor_parameters(parameters)
        previous_params_tensor = self._load_param_tensors(key)

        if not previous_params_tensor:
            ### This is the first round, only add gaussian noise

            # Generate different kinds of Gaussian noise
            noise = self._generate_adaptive_noise(current_params_tensor)
            
            # Add the noise to the current parameters
            new_params_tensor = [param + n for param, n in zip(current_params_tensor, noise)]

        else:
            ### This is after the first round, previous parameters are available!

            # Compute the delta of the weigts of the previous 
            # --- Note that we changed this from (prev - current) to (current - prev), maybe there is a mistake in Equation 2 in the paper of Lin et al. ---
            delta_weights = [current_param - prev_param for current_param, prev_param in zip(current_params_tensor, previous_params_tensor)]

            # Generate different kinds of Gaussian noise
            noise = self._generate_adaptive_noise(current_params_tensor)

            # Add the delta weights and noise to the current parameters to simulate the update
            new_params_tensor = [current_param_tensor + delta_weight + n for current_param_tensor, delta_weight, n in zip(current_params_tensor, delta_weights, noise)]

        new_params = [param.cpu().numpy() for param in new_params_tensor]

        # Set the current parameters as the previous for the next round
        self._save_params(key, parameters)

        return (new_params, 
                len(self.trainloader),
                {"partition_id": self.partition_id}
            )