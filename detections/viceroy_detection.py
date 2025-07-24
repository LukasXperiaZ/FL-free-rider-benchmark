from detections.detection import Detection
from flwr.common import parameters_to_ndarrays
import numpy as np
from detections.gradient_scaling_mitigation_algorithms.fools_gold import FoolsGold
from typing import Dict

class ViceroyDetection(Detection):
    def __init__(self, config):
        self.config = config
        self.R_t = {}
        self.H_t = {}
        self.omega = config.get("omega")
        self.eta = config.get("eta")
        self.kappa = config.get("kappa")
        self.skip_first_round = config.get("skip_first_round")
        self.free_rider_threshold = config.get("free_rider_threshold")
        self.first_round = True

    def detect(self, server_round, client_ids, client_updates, client_metrics, global_model):
        G_t_1  = parameters_to_ndarrays(global_model)

        # Sort the client_ids and client_updates
        sorted_cids = sorted(client_ids)

        cid_to_update = {}
        for cid, update in zip(client_ids, client_updates):
            cid_to_update[cid] = update
        
        sorted_updates = [cid_to_update[i] for i in sorted_cids]

        client_ids = sorted_cids
        client_updates = sorted_updates

        # Initialize the reputations and histories in the first round
        if self.first_round:
            for i in client_ids:
                self.H_t[i] = [np.zeros_like(layer) for layer in client_updates[0]]
                self.R_t[i] = 0.0

        # Estimate the client gradients by substracting the global model from the local models.
        client_gradients = []
        for update in client_updates:
            gradient_c = [w_t - w_t_1 for w_t, w_t_1 in zip(update, G_t_1)]
            client_gradients.append(gradient_c)

        # --- Viceroy detection ---
        # Update the reputation
        H_t_1, R_t_1 = self._reputation_update(client_gradients, client_ids, self.H_t, self.R_t, self.omega, self.eta)

        if self.first_round:
            # Set the reputation of the first round to 0.
            # We do this since R_t_1 depends on H_t, which is also only initialized in the first round.
            # If R_t_1 is 0, the weighted sum will only use the weights that were computed with respect to the current client gradients.
            for cid in client_ids:
                R_t_1[cid] = 0.0
        
        # Scale the reputation with the scaling values of FedScale
        # Note: In the first round, both fed_scale values will be equal, since H_t_1 is equal to the client gradients of the first round.
        fed_scale_h_t_1 = self._fed_scale(H_t_1.values(), client_ids)
        fed_scale_grad_t_1 = self._fed_scale(client_gradients, client_ids)

        p = {}
        for cid in client_ids:
            p_i = R_t_1[cid]*fed_scale_h_t_1[cid] + (1-R_t_1[cid])*fed_scale_grad_t_1[cid]
            p[cid] = p_i

        # Update H_t and R_t
        self.H_t = H_t_1
        self.R_t = R_t_1  

        if self.first_round:
            self.first_round = False
            if self.skip_first_round:
                return client_ids
        else:
            # Filter clients based on the p values to exclude free riders.
            filtered_client_ids = [cid for cid in client_ids if p[cid] > self.free_rider_threshold]
            return filtered_client_ids


    def _reputation_update(self, client_gradients, client_ids, H_t: Dict, R_t: Dict, omega, eta):
        """
        Important assumption: client_gradients, client_ids and H_t.values() is all sorted in the same way!
        """
        p_i_H = self._fed_scale(H_t.values(), client_ids)
        if self.first_round:
            # In the first round, set the historical update weights to 0
            for cid in p_i_H.keys():
                p_i_H[cid] = 0.0

        p_i_C = self._fed_scale(client_gradients, client_ids)

        for c_gradients, i in zip(client_gradients, client_ids):
            R_t[i] = max(min(R_t[i] + eta/2 * (1 - 2*np.abs(p_i_H[i] - p_i_C[i])), 1), 0)
            
            H_t[i] = [omega*h_grad + c_grad for h_grad, c_grad in zip(H_t[i], c_gradients)]

        return H_t, R_t

    def _fed_scale(self, client_gradients, client_ids):
        """
        Performs anomaly detection on the client_gradients.

        Returns: A dict mapping from client_ids to weightings
        """
        fools_gold = FoolsGold(self.kappa)
        return fools_gold.get_scaled_gradients(client_gradients, client_ids)
