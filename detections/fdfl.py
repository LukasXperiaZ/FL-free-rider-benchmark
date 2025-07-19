from detections.detection import Detection
from flwr.common import parameters_to_ndarrays
import numpy as np
import json
from sklearn.cluster import KMeans
from typing import List, Dict
import pickle

class FDFLDetection(Detection):
    def __init__(self, config):
        self.config = config

    def flatten_weights(self, weights: List[np.ndarray]) -> np.ndarray:
        """Flattens a list of NumPy arrays (model weights) into a single 1D array."""
        return np.concatenate([w.flatten() for w in weights])

    def detect(self, server_round, client_ids, client_updates, client_metrics, global_model):
        # data_to_save = {
        #     'client_ids': client_ids,
        #     'client_updates': client_updates,
        #     'client_metrics': client_metrics
        # }

        # with open("./FDFL_sample_data_10.pkl", 'wb') as f:
        #     pickle.dump(data_to_save, f)

        # raise RuntimeError("Done with saving")


        # 0. Parse the counts of labels of all clients
        label_counts_dict_list = [] 
        for client_metric in client_metrics:
            label_counts_dict = json.loads(client_metric.get("label_counts"))
            label_counts_dict_list.append(label_counts_dict)
        
        # Get a list of all labels
        label_set = set()
        for label_count_dict in label_counts_dict_list:
            label_set.update(set(label_count_dict.keys()))
        sort_label_list = sorted(list(label_set))

        # label_counts:     For each client, it contains a list with label counts ordered like sorted_label_list
        label_counts = []
        for label_count_dict in label_counts_dict_list:
            l_counts = []
            for label in sort_label_list:
                if label in label_count_dict:
                    amount = label_count_dict[label]
                else:
                    amount = 0
                l_counts.append(amount)

            label_counts.append(l_counts)
        
        # Generate a dict mapping from client_ids to label_counts
        client_ids_to_label_counts = {}
        for cid, l_counts in zip(client_ids, label_counts):
            client_ids_to_label_counts[cid] = l_counts

        # 1. Perform K-means clustering on the weights of the client models
        n_clusters = self.config.get("n_clusters")
        clients_per_cluster = self._k_means(client_ids, client_updates, n_clusters)

        # 2. For each cluster, check for each client if its submitted data distribution is similar or different to other clients.
        tau = self.config.get("tau")
        flag = {}
        kept_ids = []
        for cluster_id in clients_per_cluster.keys():
            clients = clients_per_cluster[cluster_id]
            for c_i in clients:
                flag[c_i] = []
                for c_j in clients:
                    if c_i != c_j:
                        lambda_i = client_ids_to_label_counts[c_i]
                        lambda_j = client_ids_to_label_counts[c_j]
                        alpha_i_j = self._compute_cosine_similarity(lambda_i, lambda_j)
                        
                        if alpha_i_j < tau:
                            flag[c_i].append(alpha_i_j)
                
                if len(flag[c_i]) < n_clusters-1:
                    kept_ids.append(c_i)

        return kept_ids


    def _k_means(self, client_ids, client_updates, n_clusters, random_state = 42):
        """
        Perform k-means clustering on client updates (list of model weights).

        Return: clients_per_cluster, a list containing lists of client_ids for each cluster.
        """
        # 1. Flatten and stack weights
        flat_client_updates = []
        for update in client_updates:
            flat_update = self.flatten_weights(update)
            flat_client_updates.append(flat_update)
        
        data_matrix = np.array(flat_client_updates)
        # Handle cases where all weights are zero or constant, leading to zero std
        # K-means might struggle with zero variance features.
        if np.std(data_matrix) < 1e-9:
            print("Warning: All client weights are very similar (low variance). K-means might not be meaningful.")
            clients_in_single_cluster = {0: client_ids}
            return clients_in_single_cluster
        
        # 2. Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_assignments = kmeans.fit_predict(data_matrix)

        # 3. Map Client IDs to Clusters
        clients_per_cluster: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
        for i, client_id in enumerate(client_ids):
            cluster_id = cluster_assignments[i]
            clients_per_cluster[cluster_id].append(client_id)

        return clients_per_cluster
    
    def _compute_cosine_similarity(self, list1: list, list2: list) -> float:
        """
        Compute the cosine similarity between two lists containing numerical values
        """
        if len(list1) != len(list2):
            raise ValueError("Input lists must have the same length to compute cosine similarity.")

        # Convert lists to NumPy arrays for efficient vector operations
        vec1 = np.array(list1, dtype=float)
        vec2 = np.array(list2, dtype=float)

        # Compute the dot product
        dot_product = np.dot(vec1, vec2)

        # Compute the L2 (Euclidean) norm (magnitude) of each vector
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # Handle cases where one or both vectors are zero vectors (magnitude is 0)
        # Cosine similarity is undefined in these cases, so we typically return 0.
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = dot_product / (norm_vec1 * norm_vec2)
        
        return similarity
