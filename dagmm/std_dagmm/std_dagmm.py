import torch
import torch.nn as nn
import numpy as np
from dagmm.dagmm.dagmm import DAGMM
from dagmm.dagmm.dagmm import DAGMM_Hyperparameters

class STD_DAGMM(DAGMM):
    """
    An extension of DAGMM that also considers the standard deviation of the inputs and feeds it into the GMM.
    """

    def __init__(self, device, hyperparameters: DAGMM_Hyperparameters):
        super().__init__(device, hyperparameters)
        
        latent_dim = hyperparameters.latent_dim
        estimation_hidden_size = hyperparameters.estimation_hidden_size
        n_gmm = hyperparameters.n_gmm
        self.total_latent_dim = latent_dim + 3  # account for rec_euclidean, rec_cosine and std_inputs

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
        std_inputs = torch.std(inputs, dim=1, keepdim=True)

        latents = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1), std_inputs], dim=1)

        # Note that the self.estimation network's output (gamma) doesn't directly influence the final energy score during inference, 
        #   beyond its indirect role in helping to learn the phi, mu, cov parameters during training.
        gamma = self.estimation(latents)
        return reconst, latents, gamma
    
