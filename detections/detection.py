class Detection:
    def __init__(self, config):
        raise NotImplementedError

    def detect(self, server_round, client_ids, client_updates):
        raise NotImplementedError