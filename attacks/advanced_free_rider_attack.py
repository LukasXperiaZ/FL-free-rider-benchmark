from task import set_weights
from attacks.advanced_delta_weights_attack import AdvancedDeltaWeightsAttack
import torch
import numpy as np
from flwr.common import ArrayRecord
import random

class AdvancedFreeRiderAttack(AdvancedDeltaWeightsAttack):

    def convert_tensor_list_to_numpy_list(self, tensor_list):
        """
        Converts a list of PyTorch tensors back to a list of NumPy arrays.
        """
        numpy_arrays = []
        for tensor_param in tensor_list:
            # Move tensor to CPU and convert to NumPy array
            numpy_arrays.append(tensor_param.cpu().numpy())
        return numpy_arrays

    def fit(self, parameters, config):
        print(f"fit called for AdvancedFreeRiderAttack, self ID: {id(self)}")
        set_weights(self.net, parameters)
        # TODO updating global model history (theta's) does not work

        key_theta_minus_1 = "theta_minus_1"
        key_theta_minus_2 = "theta_minus_2"
        key_t = "t"
        key_theta_0 = "theta_0"
        key_theta_1 = "theta_1"

        if key_t not in self.client_state.keys():
            ### Initialize the first round
            self.client_state[key_t] = ArrayRecord({key_t: [1]})
            # Save the initial global model
            self._save_param_tensors(key_theta_0, parameters)
        else:
            ### Update the current iteration t
            self.client_state[key_t].get(key_t)[0] += 1
        t = int(self.client_state[key_t].get(key_t)[0])

        # Gaussian noise parameters
        mu = 0
        #d = 0 #TODO change
        sigma = 1e-3    #1/d

        # Advanced free rider attack
        theta_t = self.get_tensor_parameters(parameters)
        theta_t_minus_1 = self._load_param_tensors(key_theta_minus_1)
        theta_t_minus_2 = self._load_param_tensors(key_theta_minus_2)

        if not theta_t_minus_1:
            ### This is the first round, only add gaussian noise
            # Generate Gaussian noise
            noise = [(torch.randn_like(param) + mu) * sigma for param in theta_t]

            # Add the noise to the current parameters
            new_params_tensor = [param + n for param, n in zip(theta_t, noise)]

            ### Update history
            # Set theta_minus_1 = theta
            self._save_param_tensors(key_theta_minus_1, parameters)

        elif not theta_t_minus_2:
            ### This is the second round.
            self._save_param_tensors(key_theta_1, parameters)
            # Theta-1 is available. Perform the advanced delta weights attack.
            # Compute the delta of the weigts of the previous 
            # --- Note that we changed this from (prev - current) to (current - prev), maybe there is a mistake in Equation 2 in the paper of Lin et al. ---
            delta_t_t1 = [current_param - prev_param for current_param, prev_param in zip(theta_t, theta_t_minus_1)]

            # Generate Gaussian noise
            noise = [(torch.randn_like(dw) + mu) * sigma for dw in delta_t_t1]

            # Add the delta weights and noise to the current parameters to simulate the update
            new_params_tensor = [current_param_tensor + delta_weight + n for current_param_tensor, delta_weight, n in zip(theta_t, delta_t_t1, noise)]

            ### Update history
            # Set theta_minus_2 = theta_minus_1
            self._save_param_tensors(key_theta_minus_2, self.convert_tensor_list_to_numpy_list(theta_t_minus_1))
            # Set theta_minus_1 = theta
            self._save_param_tensors(key_theta_minus_1, parameters)

        else:
            ### This is after the second round, theta-1 and theta-2 are available!
            delta_t_t1 = [current_param - prev_param for current_param, prev_param in zip(theta_t, theta_t_minus_1)]
            delta_t1_t2 = [current_param - prev_param for current_param, prev_param in zip(theta_t_minus_1, theta_t_minus_2)]
            
            dalta_t_t1_flat = torch.cat([t.flatten() for t in delta_t_t1])
            delta_t_t2_flat = torch.cat([t.flatten() for t in delta_t1_t2])
            
            delta_t_t1_l2 = dalta_t_t1_flat.norm(p=2).item()
            delta_t1_t2_l2 = delta_t_t2_flat.norm(p=2).item()
            factor = delta_t_t1_l2 / delta_t1_t2_l2

            # print(factor) Starts at ~0.4 and converges to ~0.9

            U_f_theta = [tensor * factor for tensor in delta_t_t1]

            ## Compute parameters for Gaussian Noise
            l_t = delta_t_t1_l2

            theta_0 = self._load_param_tensors(key_theta_0)
            theta_1 = self._load_param_tensors(key_theta_1)
            delta_0_1 = [current_param - prev_param for current_param, prev_param in zip(theta_1, theta_0)]
            delta_0_1_flat = torch.cat([t.flatten() for t in delta_0_1])
            l_1 = delta_0_1_flat.norm(p=2).item()

            lambda_ = np.log((l_t/l_1) ** (1/(t-1)))
            C = 0.5 # TODO find a good value for C.
            E_cos_beta = C**2 / (C**2 + np.e**(2*lambda_*t))
            n = self.config.get("n")
            U_f_theta_flat = torch.cat([t.flatten() for t in U_f_theta])
            U_f_theta_l2 = U_f_theta_flat.norm(p=2).item()
            # Calculate |φ(t)|
            phi_t_l2 = np.sqrt(n**2 / (n + (n**2 - n)*E_cos_beta) - 1) * U_f_theta_l2

            # HYPERPARAMETER: Percentage of parameters to add Gaussian noise to.
            d = 0.1 # TODO find a good value for d.
            n_parameters = 0
            for tensor in U_f_theta:
                n_params = tensor.numel()
                n_parameters += n_params

            n_params_subset = d * n_parameters
            std = 1/n_params_subset
            # TODO compute gaussian noise to subset of parameters, i.e. go over every parameter and if the following check passes, add the gaussian noise (|phi(t)|*N(0, std)).
            if random.random() < d:
                pass

            # Update tensor
            new_params_tensor = [current_param_tensor + delta_weight for current_param_tensor, delta_weight in zip(theta_t, U_f_theta)]

            ### Update history
            # Set theta_minus_2 = theta_minus_1
            self._save_param_tensors(key_theta_minus_2, self.convert_tensor_list_to_numpy_list(theta_t_minus_1))
            # Set theta_minus_1 = theta
            self._save_param_tensors(key_theta_minus_1, parameters)

        new_params = [param.cpu().numpy() for param in new_params_tensor]
        return (new_params, 
                len(self.trainloader),
                {"partition_id": self.partition_id}
            )