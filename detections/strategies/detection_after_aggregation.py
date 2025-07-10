from strategy import FedAvgWithDetections
from flwr.common import logger, parameters_to_ndarrays

class FedAvgWithDetectionsAfterAggregation(FedAvgWithDetections):

    def aggregate_fit(self, server_round, results, failures):

        ### First aggreagte
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        ### Then provide the detection the aggregated global model.
        self.detection_handler.set_aggregated_global_model(parameters_aggregated)

        ### Then perform the detection
        client_updates = []
        client_ids = []
        flower_cid_to_partition_id = {}
        partition_id_to_flower_cid = {}

        for client_proxy, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            cid = client_proxy.cid
            client_updates.append(weights)
            client_ids.append(cid)
            # Ensure 'partition_id' is in metrics, provide a fallback if not
            partition_id = fit_res.metrics.get("partition_id", "UNKNOWN_PARTITION_ID")
            flower_cid_to_partition_id[cid] = partition_id
            partition_id_to_flower_cid[partition_id] = cid
  
        partition_ids = [flower_cid_to_partition_id[cid] for cid in client_ids]
        # Detect anomalies
        kept_partition_ids = self.detection_handler.detect_anomalies(server_round, partition_ids, client_updates, self.global_model)
        kept_client_ids = [partition_id_to_flower_cid[partition_id] for partition_id in kept_partition_ids]

        # Determine dropped clients
        dropped_ids = set(client_ids) - set(kept_client_ids)
        self.banned_client_ids.update(dropped_ids)

        dropped_partition_ids = set([flower_cid_to_partition_id[cid] for cid in dropped_ids])
        self.banned_partition_ids.update(dropped_partition_ids)

        assert len(dropped_ids) == len(dropped_partition_ids)

        if dropped_partition_ids:
            print(f"[Round {server_round}] Dropped clients (anomalous):\t{sorted(list(dropped_partition_ids))}")
        else:
            print(f"[Round {server_round}] No clients dropped.")

        print(f"All anomalous clients detected and removed:\t{sorted(list(self.banned_partition_ids))}")

        # Update the global model        
        self.global_model = parameters_aggregated

        # TODO CHANGE
        return parameters_aggregated, metrics_aggregated