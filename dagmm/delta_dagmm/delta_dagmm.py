import torch
import torch.nn as nn
from dagmm.dagmm.dagmm import DAGMM
from dagmm.dagmm.dagmm import DAGMM_Hyperparameters

class DELTA_DAGMM(DAGMM):
    """
    An extension of DAGMM that uses the difference between local and global model weights (last layer) as input for DAGMM.
    Also uses the mean of the input as an additional parameter for the GMM.
    """
    def __init__(self, device, hyperparameters: DAGMM_Hyperparameters):
        super().__init__(device, hyperparameters)
        
        latent_dim = hyperparameters.latent_dim
        estimation_hidden_size = hyperparameters.estimation_hidden_size
        n_gmm = hyperparameters.n_gmm
        self.total_latent_dim = latent_dim + 3  # account for rec_euclidean, rec_cosine and mean_inputs

        # Estimation network
        self.estimation = nn.Sequential(
            nn.Linear(self.total_latent_dim, estimation_hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(estimation_hidden_size, n_gmm),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        enc = self.encoder(inputs)
        reconst = self.decoder(enc)

        rec_euclidean, rec_cosine = self.compute_reconstruction(inputs, reconst)
        mean_inputs = torch.mean(inputs, dim=1, keepdim=True)

        latents = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1), mean_inputs], dim=1)

        # Note that the self.estimation network's output (gamma) doesn't directly influence the final energy score during inference, 
        #   beyond its indirect role in helping to learn the phi, mu, cov parameters during training.
        gamma = self.estimation(latents)
        return reconst, latents, gamma