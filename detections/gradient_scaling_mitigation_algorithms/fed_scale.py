class FedScale():
    """
    An abstract class, from which subclasses can be derived to be used as a component of Viceroy.
    """
    
    def get_scaled_gradients(self, client_gradients, client_ids):
        raise NotImplementedError("This method is not implemented yet!")