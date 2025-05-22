import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DAGMM(nn.Module):
    """
    DAGMM, an anomaly detector.

    It uses the AE for representation learning and reconstruction error, 
    and an estimation network to help the GMM parameters (phi, mu, cov) converge to a good fit for the 'normal' z space. 
    The final energy is then computed via the overall likelihood under that learned GMM.
    """
    def __init__(self, input_dim=62006, latent_dim=16, n_gmm=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 1024),
            nn.Tanh(),
            nn.Linear(1024, input_dim)
        )

        self.total_latent_dim = latent_dim + 2  # account for rec_euclidean and rec_cosine

        self.estimation = nn.Sequential(
            nn.Linear(self.total_latent_dim, 10),
            nn.Tanh(),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1)
        )

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, self.total_latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, self.total_latent_dim, self.total_latent_dim))

    def euclidean_distance(self, a, b):
        return (a - b).norm(p=2, dim=1)

    def forward(self, x):
        enc = self.encoder(x)
        x_rec = self.decoder(enc)

        rec_cosine = 1 - F.cosine_similarity(x, x_rec, dim=1)
        rec_euclidean = self.euclidean_distance(x, x_rec)

        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

        # Note that the self.estimation network's output (gamma) doesn't directly influence the final energy score during inference, 
        #   beyond its indirect role in helping to learn the phi, mu, cov parameters during training.
        gamma = self.estimation(z)
        return enc, x_rec, z, gamma

    def compute_gmm_params(self, z, gamma):
        eps = 1e-10
        N, D = z.size()
        K = gamma.size(1)

        sum_gamma = gamma.sum(dim=0)  # [K]
        phi = sum_gamma / (N + eps)

        mu = (gamma.unsqueeze(2) * z.unsqueeze(1)).sum(dim=0) / (sum_gamma.unsqueeze(1) + eps)  # [K, D]

        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)  # [N, K, D]
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)  # [N, K, D, D]

        cov = (gamma.unsqueeze(2).unsqueeze(3) * z_mu_outer).sum(dim=0) / (sum_gamma.view(-1, 1, 1) + eps)  # [K, D, D]

        self.phi.data.copy_(phi)
        self.mu.data.copy_(mu)
        self.cov.data.copy_(cov)

        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        device = z.device
        eps = 1e-6

        if phi is None: phi = self.phi
        if mu is None: mu = self.mu
        if cov is None: cov = self.cov

        K, D, _ = cov.size()
        N = z.size(0)

        z = z.unsqueeze(1)  # [N, 1, D]
        mu = mu.unsqueeze(0)  # [1, K, D]
        diff = z - mu  # [N, K, D]

        # Precompute inverse and determinant
        cov_inv = torch.inverse(cov + torch.eye(D, device=device) * eps)  # [K, D, D]
        log_det_cov = torch.logdet(cov + torch.eye(D, device=device) * eps)  # [K]

        # Mahalanobis distance: [N, K]
        maha = torch.einsum('nkd,kde,nke->nk', diff, cov_inv, diff)

        # Log Gaussian probability: [N, K]
        log_prob = -0.5 * (maha + D * np.log(2 * np.pi) + log_det_cov.unsqueeze(0))  # [N, K]
        weighted_log_prob = log_prob + torch.log(phi.unsqueeze(0) + eps)  # [N, K]

        # Log-sum-exp over components for each sample
        energy = -torch.logsumexp(weighted_log_prob, dim=1)  # [N]

        # Optional diagnostics
        cov_diag = torch.sum(torch.stack([1.0 / (cov[i].diagonal() + eps).sum() for i in range(K)]))

        if size_average:
            energy = energy.mean()

        return energy, cov_diag


    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = F.mse_loss(x_hat, x)

        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag
