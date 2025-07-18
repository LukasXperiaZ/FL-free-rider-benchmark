from detections.detection import DetectionAfterAggregation
from flwr.common import parameters_to_ndarrays
import numpy as np

class FGFLDetection(DetectionAfterAggregation):
    def __init__(self, config):
        self.config = config
        self.aggregated_global_model = None

    def set_aggregated_global_model(self, aggregated_global_model):
        self.aggregated_global_model = aggregated_global_model

    def detect(self, server_round, client_ids, client_updates, client_metrics, global_model):
        G_t_1  = parameters_to_ndarrays(global_model)

        if not self.aggregated_global_model:
            raise RuntimeError("'set_aggregated_global_model' has to be called before detect() is called!")
        G_t = parameters_to_ndarrays(self.aggregated_global_model)

        # Represents G tidal of the paper
        gradient_G = [w_t - w_t_1 for w_t, w_t_1 in zip(G_t, G_t_1)]

        gradients_clients = []
        for update in client_updates:
            gradient_c = [w_t - w_t_1 for w_t, w_t_1 in zip(update, G_t_1)]
            gradients_clients.append(gradient_c)

        # Compute the gradient_distances (b_i)
        gradient_distances = []
        for gradient in gradients_clients:
            gradient_dist = self._square_euclidean_norm(gradient, gradient_G)
            gradient_distances.append(gradient_dist)

        # Compute the 0 gradient (G_0)
        gradient_G_0 = [np.zeros_like(layer_array) for layer_array in gradient_G]
        # Compute the threshold (b_h)
        b_h = self._square_euclidean_norm(gradient_G, gradient_G_0)

        # Compute the contributions (C_i)
        contributions = []
        for b_i in gradient_distances:
            contribution = 1 - (b_i/b_h)
            #print(b_i, b_h, contribution)
            contributions.append(contribution)

        # Identify free riders
        kept_ids = []
        for contribution, client_id in zip(contributions, client_ids):
            if contribution > -2:
                kept_ids.append(client_id)
        return kept_ids


    def _square_euclidean_norm(self, gradients_1, gradients_2):
        if len(gradients_1) != len(gradients_2):
            raise ValueError("Gradients must have the same shape.")
        
        squared_norm_sum = 0.0
        for i in range(len(gradients_1)):
            layer_grad_1 = gradients_1[i]
            layer_grad_2 = gradients_2[i]

            if layer_grad_1.shape != layer_grad_2.shape:
                raise ValueError(f"Layer {i} has mismatched shapes: {layer_grad_1.shape} vs {layer_grad_2.shape}")

            # Compute the difference between the layer gradients
            difference = layer_grad_1 - layer_grad_2

            # Compute the squared Euclidean norm for this layer
            squared_norm_sum += np.sum(difference**2)

        return squared_norm_sum