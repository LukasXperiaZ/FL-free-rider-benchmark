from strategy import FedAvgWithDetections
from logging import WARNING
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
import numpy as np
from functools import partial, reduce

class RFFL(FedAvgWithDetections):
    """
    A class that behaves like FedAvgWithDetections but implements the custom aggregation logic of RFFL.
    """

    def __init__(self, run_config, use_wandb, detection_handler, alpha, beta, gamma, *args, **kwargs):
        super().__init__(run_config, use_wandb, detection_handler, *args, **kwargs)

        # reputations[i] returns the reputation of client i of the past round (t-1)
        # The sum of all reputations of all reputable clients (reputations[i] >= beta) is always 1.
        self.reputations = {}
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_fit(self, server_round, results, failures):
        client_ids = []
        flower_cid_to_partition_id = {}
        partition_id_to_flower_cid = {}

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            client_ids.append(cid)
            # Ensure 'partition_id' is in metrics, provide a fallback if not
            partition_id = fit_res.metrics.get("partition_id", "UNKNOWN_PARTITION_ID")
            flower_cid_to_partition_id[cid] = partition_id
            partition_id_to_flower_cid[partition_id] = cid

        ### 1. Perform weighted aggregation
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        
        if not self.reputations:
            # This is the first round
            # Generate the initial reputations (1/n_clients).
            for _, fit_res in results:
                i = fit_res.metrics.get("partition_id")
                self.reputations[i] = 1/len(results)
           
        weights_reputations = [(parameters_to_ndarrays(fit_res.parameters), self.reputations[fit_res.metrics.get("partition_id")]) for _, fit_res in results]
        aggregated_ndarrays = self.aggregate(weights_reputations)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        ### 2. Update Reputations and remove low ones
        dropped_client_ids = []
        dropped_partition_ids = []
        w_g_t = aggregated_ndarrays
        for _, fit_res in results:
            i = fit_res.metrics.get("partition_id")
            w_i_t = parameters_to_ndarrays(fit_res.parameters)
            r_i_t = self.calculate_cosine_similarity(w_g_t, w_i_t)
            self.reputations[i] = self.alpha*self.reputations[i] + (1 - self.alpha)*r_i_t

            if self.reputations[i] < self.beta:
                # Remove client from the reputation list
                del self.reputations[i]
                # Remove the client from FL
                banned_cid = partition_id_to_flower_cid[i]
                dropped_client_ids.append(banned_cid)
                dropped_partition_ids.append(i)

        self.banned_client_ids.update(set(dropped_client_ids))

        if dropped_partition_ids:
            print(f"[Round {server_round}] Dropped clients (anomalous):\t{sorted(list(dropped_partition_ids))}")
        else:
            print(f"[Round {server_round}] No clients dropped.")

        self.banned_partition_ids.update(dropped_partition_ids)
        print(f"All anomalous clients detected and removed:\t{sorted(list(self.banned_partition_ids))}")
        
        ### 3. Normalize Reputations to satisfy Sum(reputations)=1
        rep_sum = sum(self.reputations.values())
        for i in self.reputations:
            self.reputations[i] /= rep_sum
        

        # (Behave like FedAvgWithDetections) Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
        

    def aggregate(self, weights_reputations):
        weighted_normalized_updates = []

        for weights, reputation in weights_reputations:
            # weights represents Δw_i^(t) for a single client.
            # It is a list of numpy arrays, where each array corresponds to a layer's weights.

            # We do not use normalization as the paper specifies as we aggregate the models themselfs instead of the gradients.
            # Therefore, we would normalizing whole models which is not reasonable. 
            normalization = 1   

            # Create a list of weights, multiplied by the reputation (between 0 and 1, the reputations already sum up to 1)
            weighted_weights = [reputation * layer * normalization for layer in weights]
            weighted_normalized_updates.append(weighted_weights)

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) for layer_updates in zip(*weighted_normalized_updates)
        ]
        return weights_prime
    
    def calculate_cosine_similarity(self, model1: NDArrays, model2: NDArrays, epsilon: float = 1e-10) -> float:
        """
        Calculates the cosine similarity between two sets of model parameters.
        Both sets of parameters are expected to be lists of numpy arrays.

        Args:
            model1: The first set of model parameters.
            model2: The second set of model parameters.
            epsilon: A small value added to the denominator to prevent division by zero
                    if any parameter vector has a zero norm.

        Returns:
            The cosine similarity as a float.
        """
        flat_model_1 = np.concatenate([layer.flatten() for layer in model1])
        flat_model_2 = np.concatenate([layer.flatten() for layer in model2])

        dot_product = np.dot(flat_model_1, flat_model_2)

        norm_model_1 = np.linalg.norm(flat_model_1)
        norm_model_2 = np.linalg.norm(flat_model_2)

        # Calculate cosine similarity
        denominator = norm_model_1 * norm_model_2
        if denominator < epsilon:   # The denominator is practically 0
            return 0.0
        else:
            cosine_sim = dot_product / denominator
            return cosine_sim
