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
        # Flatten the gradients and store them in a single NumPy array
        flattened_gradients = np.array([np.concatenate([w.flatten() for w in grad_i]) for grad_i in client_gradients])

        # First step: compute the cosine similarities

        # Calculate the dot product of all gradient pairs (numerator)
        dot_product_matrix = np.dot(flattened_gradients, flattened_gradients.T)

        # Calculate the L2 norm of each flattened gradient vector
        norm_vector = np.linalg.norm(flattened_gradients, axis=1)

        # Create an outer product of the norm vector to get the matrix of products of norms (denominator)
        norm_product_matrix = np.outer(norm_vector, norm_vector)

        # Add a small epsilon to avoid division by zero
        norm_product_matrix[norm_product_matrix == 0] = 1e-9

        # Compute the cosine similarity matrix by element-wise division
        cosine_similarity_matrix = dot_product_matrix / norm_product_matrix

        # Extract the maximum similarity for each client (v)
        # We need to set the diagonal to a very low number to ignore self-similarity
        np.fill_diagonal(cosine_similarity_matrix, -1)
        v = np.max(cosine_similarity_matrix, axis=1)

        v_dict = {client_id: val for client_id, val in zip(client_ids, v)}
        cs_i_j = {client_ids[i]: {client_ids[j]: cosine_similarity_matrix[i, j] for j in range(len(client_ids))} for i in range(len(client_ids))}

        # Second step: scale the cosine similarities and initialize the weighting a
        a = {}
        for i in client_ids:
            for j in client_ids:
                if i != j:
                    if v_dict[j] > v_dict[i]:
                        cs_i_j[i][j] = cs_i_j[i][j] * v_dict[i] / v_dict[j]
            
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
