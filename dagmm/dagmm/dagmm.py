import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import os

EPSILON = np.finfo(np.float32).eps

# A floor value for stability in computing inverse of covariance matrix of GMM.
COV_EPS = 1.0e-6  # a magic number

class DAGMM_Hyperparameters:
    def __init__(self, dimensions, latent_dim, estimation_hidden_size, n_gmm):
        self.dimensions = dimensions
        self.latent_dim = latent_dim
        self.estimation_hidden_size = estimation_hidden_size
        self.n_gmm = n_gmm

    def save_params(self, filepath):
        """
        Saves the hyperparameters to a YAML file.

        Args:
            filepath (str): The path to the YAML file where parameters will be saved.
        """
        params_dict = {
            "dimensions": self.dimensions,
            "latent_dim": self.latent_dim,
            "estimation_hidden_size": self.estimation_hidden_size,
            "n_gmm": self.n_gmm
        }
        with open(filepath, 'w') as file:
            yaml.dump(params_dict, file, default_flow_style=False)

    @classmethod
    def load_params(cls, filepath):
        """
        Loads hyperparameters from a YAML file and creates a new
        DAGMM_Hyperparameters instance.

        Args:
            filepath (str): The path to the YAML file from which parameters will be loaded.

        Returns:
            DAGMM_Hyperparameters: A new instance of the class with loaded parameters.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Hyperparameters file not found at: {filepath}")

        with open(filepath, 'r') as file:
            params_dict = yaml.safe_load(file)
        
        # Instantiate the class with the loaded parameters
        loaded_instance = cls(
            dimensions=params_dict.get("dimensions"),
            latent_dim=params_dict.get("latent_dim"),
            estimation_hidden_size=params_dict.get("estimation_hidden_size"),
            n_gmm=params_dict.get("n_gmm")
        )
        return loaded_instance

    def __repr__(self):
        return (f"DAGMM_Hyperparameters(dimensions={self.dimensions}, "
                f"latent_dim={self.latent_dim}, "
                f"estimation_hidden_size={self.estimation_hidden_size}, "
                f"n_gmm={self.n_gmm})")

class DAGMM(nn.Module):
    """
    DAGMM, an anomaly detector.

    It uses the AE for representation learning and reconstruction error, 
    and an estimation network to help the GMM parameters (phi, mu, cov) converge to a good fit for the 'normal' z space. 
    The final energy is then computed via the overall likelihood under that learned GMM.
    """
    def __init__(self, device, hyperparameters: DAGMM_Hyperparameters):
        # Make sure that the first entry in dimensions is your input dimension!
        dimensions = hyperparameters.dimensions
        latent_dim = hyperparameters.latent_dim
        estimation_hidden_size = hyperparameters.estimation_hidden_size
        n_gmm = hyperparameters.n_gmm

        super().__init__()
        self.device = device

        # Encoder layers
        encoder_layers = []
        for i in range(0, len(dimensions)-1):
            encoder_layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
            encoder_layers.append(nn.Tanh())
        encoder_layers.append(nn.Linear(dimensions[-1], latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        dimensions_reverse = dimensions[::-1]
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, dimensions_reverse[0]))
        for i in range(0, len(dimensions_reverse)-1):
            decoder_layers.append(nn.Linear(dimensions_reverse[i], dimensions_reverse[i+1]))
            if i < len(dimensions_reverse)-2:
                # Only add Tanh activation function after the hidden layers
                decoder_layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*decoder_layers)

        self.total_latent_dim = latent_dim + 2  # account for rec_euclidean and rec_cosine

        # Estimation network
        self.estimation = nn.Sequential(
            nn.Linear(self.total_latent_dim, estimation_hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(estimation_hidden_size, n_gmm),
            nn.Softmax(dim=1)
        )

    def compute_reconstruction(self, inputs, reconst):
        rec_euclidean = (inputs - reconst).norm(2, dim=1) / torch.clamp(
            inputs.norm(2, dim=1), min=EPSILON
        )
        rec_cosine = F.cosine_similarity(inputs, reconst, dim=1)
        return rec_euclidean, rec_cosine

    def forward(self, inputs):
        enc = self.encoder(inputs)
        reconst = self.decoder(enc)

        rec_euclidean, rec_cosine = self.compute_reconstruction(inputs, reconst)

        latents = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

        # Note that the self.estimation network's output (gamma) doesn't directly influence the final energy score during inference, 
        #   beyond its indirect role in helping to learn the phi, mu, cov parameters during training.
        gamma = self.estimation(latents)
        return reconst, latents, gamma
    
    def compute_loss(self, inputs, lambda_energy, lambda_cov):
        reconst, latents, gamma = self.forward(inputs)

        # squared L2-norm in accordance with the original paper.
        reconst_loss = torch.mean((inputs - reconst) ** 2)

        sample_energy, cov_diag = self.compute_energy(latents, gamma)

        # objective function as Eq.(7) in the original paper.
        loss = reconst_loss + lambda_energy * sample_energy + lambda_cov * cov_diag

        loss_dict = {
            "reconst": reconst_loss.item(),
            "energy": sample_energy,
            "cov_diag": cov_diag,
        }

        return loss, loss_dict
    
    def compute_energy(self, latents, gamma, params=None, sample_mean=True):
        """Compute the sample energy function.
        N: number of samples (= size of minibatch)
        K: number of Gaussian mixture components
        D: latent dimension
        Parameters
        ----------
        latents : Torch Tensor of shape (N, D)
            Output of compression network with reconstruction features.
            This corresponds to z in Eq.(3) of the DAGMM paper.
        gamma : Torch Tensor of shape (N, K)
            Soft mixture-component membership prediction.
        params : dictionary of Torch Tensor, optional (default=None)
            Statistics of GMM for all componets.
            "mixuture" : mixture probability of shape (K, K)
            "mean" : mean vector of shape (K, D)
            "cov" : covariance matrix of shape (K, D, D)
            If this is set to None, it is newly computed from latents and gamma.
        sample_mean : bool, optional (default=True)
            If True, sample eneargy is averaged over samples.
        Returns
        ----------
        sample_energy : Torch Tensor; shape (N) or (1)
            Sample energy function.
            This corresponds to z in Eq.(6) of the DAGMM paper.
        cov_diag :
            The inverse of the diagonal entries in convariance matrix.
        """

        if params is None:
            params = self.compute_sample_stats(latents, gamma)

        # number of Gaussian mixture components
        n_mix = gamma.shape[1]

        n_dim = latents.shape[1]  # D

        z_mu = latents.unsqueeze(1) - params["mean"].unsqueeze(0)  # [N, K, D]
        z_mu = z_mu.double()

        workspace = {
            "cov_ks": [],
            "det_cov": [],
            "cov_diag": 0.0,
        }

        # [D, D]
        cov_eps = (torch.eye(params["mean"].shape[1]) * COV_EPS).to(self.device)

        # cov : [K, D, D]
        for k in range(n_mix):
            cov_k = params["cov"][k].double() + cov_eps  # cov[k] : [D, D]
            workspace["cov_ks"].append(cov_k.unsqueeze(0))  # [1, D, D]
            workspace["det_cov"].append(torch.linalg.det(cov_k).unsqueeze(0))
            workspace["cov_diag"] += torch.sum(1 / cov_k.diag())  # for regularization

        workspace["det_cov"] = torch.cat(workspace["det_cov"])  # [K]
        workspace["cov_ks"] = torch.cat(workspace["cov_ks"], dim=0)  # [K, D, D]

        # A : [K, D, D] <- workspace["cov_ks"]
        # B : [K, D, N] <- z_mu.permute(1, 2, 0)
        # AX = B -> X = A^-1 * B : [K, D, N]
        sol = torch.linalg.solve(
            workspace["cov_ks"], z_mu.permute(1, 2, 0)
        )  # [K, D, N]

        sol = sol.permute(2, 0, 1)  # [N, K, D]

        # workspace to compute sample energy function
        energy = {"quad_form": None, "weight": None}

        # z_mu : [N, K, D]
        # sol : [N, K, D]
        # quad_form : [N, K]
        energy["quad_form"] = torch.sum(z_mu * sol, dim=2)

        # phi / √|2πΣ| -> exp(log(phi / √|2πΣ|))
        # weight : [1, K]
        energy["weight"] = torch.log(params["mixture"].unsqueeze(0) + EPSILON)
        energy["weight"] -= torch.log(
            (torch.sqrt(workspace["det_cov"] + EPSILON)).unsqueeze(0) + EPSILON
        )
        energy["weight"] -= 0.5 * n_dim * torch.log(torch.tensor(2 * np.pi))

        # sample_energy : [N]
        sample_energy = -torch.logsumexp(
            -0.5 * energy["quad_form"] + energy["weight"], dim=1
        )
        if sample_mean is True:
            sample_energy = torch.mean(sample_energy)  # [1]

        return sample_energy, workspace["cov_diag"]

    @classmethod
    def compute_sample_stats(cls, latents, gamma):
        """Compute GMM statistics for sample energy function.
        This method is used only in self._compute_energy().
        N: number of samples (= size of minibatch)
        K: number of Gaussian mixture components
        D: latent dimension
        Parameters
        ----------
        latents : Torch Tensor of shape (N, D)
            Output of compression network with reconstruction features.
            This corresponds to z in Eq.(3) of the DAGMM paper.
        gamma : Torch Tensor of shape (N, K)
            Soft mixture-component membership prediction.
        Returns
        ----------
        stats : dictionary of Torch Tensor
            Statistics of GMM for all componets.
            "mixuture" : mixture probability of shape (K, K)
            "mean" : mean vector of shape (K, D)
            "cov" : covariance matrix of shape (K, D, D)
        """

        # K: number of Gaussian mixture components
        # N: number of samples (= size of minibatch)
        # D: latent dimension

        # latents : [N, D]
        # gamma : [N, K]

        # phi : [K, K]
        # gamma.size(0) : N
        phi = torch.sum(gamma, dim=0) / gamma.size(0)

        # mean : [K, D]
        # latents.unsqueeze(1) : [N, 1, D]
        # gamma.unsqueeze(-1) : [N, K, 1]
        mean = torch.sum(latents.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mean /= torch.sum(gamma, dim=0).unsqueeze(-1)

        # z_mu : [N, K, D]
        # latents.unsqueeze(1) : [N, 1, D]
        # mean.unsqueeze(0) : [1, K, D]
        z_mu = latents.unsqueeze(1) - mean.unsqueeze(0)

        # z_mu_z_mu_t : [N, K, D, D]
        # z_mu.unsqueeze(-1) : [N, K, D, 1]
        # z_mu.unsqueeze(-2) : [N, K, 1, D]
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # cov : [K, D, D]
        # gamma.unsqueeze(-1).unsqueeze(-1) : # [N, K, 1, 1]
        # z_mu_z_mu_t : [N, K, D, D]
        cov = torch.sum(
            gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0
        )  # numerator [K, D, D]
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)
        # torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1) : [K, 1, 1]

        # mixture probability, mean vector, covariace matrix
        stats = {"mixture": phi, "mean": mean, "cov": cov}

        return stats
    
    def compute_gmm_params(self, train_loader):
        self.eval()
        gmm_params = {"mixture": None, "mean": None, "cov": None}
        gamma_sum = 0
        mu_sum = 0
        cov_sum = 0
        with torch.no_grad():
            for batch in train_loader:
                inputs = batch[0].to(self.device)
                _, latents, gamma = self.forward(inputs)
                stats = self.compute_sample_stats(latents, gamma)
                batch_gamma_sum = torch.sum(gamma, dim=0)
                gamma_sum += batch_gamma_sum
                mu_sum += stats["mean"] * batch_gamma_sum.unsqueeze(-1)
                cov_sum += stats["cov"] * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)

        gmm_params["mixture"] = gamma_sum / len(train_loader)
        gmm_params["mean"] = mu_sum / gamma_sum.unsqueeze(-1)
        gmm_params["cov"] = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        return gmm_params
    
    def compute_energies(self, data_loader, gmm_params):
        self.eval()

        energies = []
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].to(self.device)
                _, latents, gamma = self.forward(inputs)
                sample_energy, _ = self.compute_energy(latents, gamma, gmm_params, sample_mean=False)
                energy = sample_energy.to("cpu").detach().numpy().copy()
                energies.append(energy)

        energies = np.concatenate(energies)
        return energies
