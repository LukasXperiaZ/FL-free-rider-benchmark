from detections.gradient_scaling_mitigation_algorithms.fed_scale import FedScale
import numpy as np

class FoolsGold(FedScale):
    def __init__(self, kappa):
        self.kappa = kappa
    """
    Implements the core parts of the FoolsGold anomaly detection algorithm.

    The original implementation is at: https://github.com/DistributedML/FoolsGold/blob/master/ML/code/model_aggregator.py
    """
    def get_scaled_gradients(self, client_gradients, client_ids, debug=False):
        # First step: compute the cosine similarities
        v = {}
        cs_i_j = {}
        for grad_i, i in zip(client_gradients, client_ids):
            flattened_grad_i = np.concatenate([w.flatten() for w in grad_i])

            cs_i_j[i] = {}
            for grad_j, j in zip(client_gradients, client_ids):
                if i != j:
                    flattened_grad_j = np.concatenate([w.flatten() for w in grad_j])
                    # Since we do not have feature importances, we calculate an unweighted cosine similarity.
                    cosine_similarity_i_j = self._cosine_similarity(flattened_grad_i, flattened_grad_j)
                    cs_i_j[i][j] = cosine_similarity_i_j

            v[i] = max(cs_i_j[i].values())

        # Second step: scale the cosine similarities and initialize the weighting a
        a = {}
        for grad_i, i in zip(client_gradients, client_ids):
            for grad_j, j in zip(client_gradients, client_ids):
                if i != j:
                    if v[j] > v[i]:
                        cs_i_j[i][j] = cs_i_j[i][j] * v[i] / v[j]
            
            # Clipping according to the official implementation
            a[i] = max(min(1 - max(cs_i_j[i].values()), 1), 0)

        # Third step: Scale the weighting a
        max_i_alpha = max(a.values()) + 1e-9
        for i in a.keys():
            a[i] = a[i]/max_i_alpha
            if a[i] == 1:
                a[i] = .99
            
            # Logit function
            a[i] = self.kappa * (np.log((a[i]/(1-a[i])) + 1e-9) + 0.5)

            # Clip to be between 0 and 1
            a[i] = max(min(a[i], 1), 0)

        if debug:
            return a, cs_i_j
        
        return a


    def _cosine_similarity(self, flattened_grad_i, flattened_grad_j):
        dot_product = np.dot(flattened_grad_i, flattened_grad_j)

        norm_vec1 = np.linalg.norm(flattened_grad_i)
        norm_vec2 = np.linalg.norm(flattened_grad_j)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
    
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity