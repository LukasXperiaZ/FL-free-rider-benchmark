class Detection:
    def __init__(self, config):
        raise NotImplementedError

    
    def detect(self, server_round, client_ids, client_updates, global_model) -> list:
        """
        Function that detects free riders.
        Returns a list of client ids that are benign (i.e. it excludes identified free riders).
        """
        # global_model: The global model which the clients started training locally from.
        raise NotImplementedError
    
class DetectionAfterAggregation(Detection):
    def set_aggregated_global_model(self, aggregated_global_model):
        raise NotImplementedError