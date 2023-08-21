import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np
from umap import UMAP
from hdbscan.flat import HDBSCAN_flat

from common_utils import reparametrize, cluster_accuracy, add_gaussian_noise, compute_metrics


class EncoderBlock(nn.Module):
    def __init__(self, channel_dims):
        super().__init__()
        self.channel_dims = channel_dims
        self.in_flatten = nn.Flatten()
        self.fc_layers = []
        for i in range(len(self.channel_dims) - 1):
            self.fc_layers.append(nn.Linear(self.channel_dims[i], self.channel_dims[i + 1]))
            if i != len(self.channel_dims) - 2:
                self.fc_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        x = self.in_flatten(x)
        x = self.fc_layers(x)

        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, channel_dims, output_shape):
        super().__init__()
        self.channel_dims = channel_dims
        self.fc_layers = []
        for i in range(len(self.channel_dims) - 1):
            if i != 0:
                self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Linear(self.channel_dims[i], self.channel_dims[i + 1]))
        self.fc_layers = nn.Sequential(*self.fc_layers)
        self.out_reshape = nn.Unflatten(1, output_shape)

    def forward(self, x):
        x = self.fc_layers(x)
        x = self.out_reshape(x)

        return x
    

class AE(nn.Module):
    def __init__(self, channel_dims, output_shape):
        super().__init__()
        self.encoder = EncoderBlock(channel_dims)
        self.decoder = DecoderBlock(channel_dims[::-1], output_shape)

    def encode(self, x):
        x = self.encoder(x)

        return x
    
    def decode(self, z):
        x = self.decoder(z)

        return x

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)

        return x
    

class VAE(nn.Module):
    def __init__(self, latent_dim, channel_dims, output_shape):
        super().__init__()
        self.encoder = EncoderBlock(channel_dims)
        self.fc_mu = nn.Linear(channel_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(channel_dims[-1], latent_dim)
        self.fc_decode = nn.Linear(latent_dim, channel_dims[-1])
        self.decoder = DecoderBlock(channel_dims[::-1], output_shape)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
    
    def decode(self, z):
        z = self.fc_decode(z)
        x = self.decoder(z)

        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparametrize(mu, logvar)
        x = self.decode(z)

        return x, mu, logvar
    

class Clusterer(nn.Module):
    def __init__(self, latent_dim, n_classes, alpha=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.alpha = alpha
        self.cluster_centers = Parameter(torch.FloatTensor(self.n_classes, self.latent_dim).fill_(0), requires_grad=True)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
    

class DEC(nn.Module):
    def __init__(self, latent_dim, n_classes, a_encoder, alpha=1.0, device=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.alpha = alpha
        self.device = device
        self.a_encoder = a_encoder.encoder
        self.clust_layer = Clusterer(latent_dim, n_classes, alpha)
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = torch.nn.Sequential(self.a_encoder, self.clust_layer)

    def init_cluster_centers(self, dataloader):
        kmeans = KMeans(n_clusters=self.n_classes, n_init=100)

        zs, ts, xs = [], [], []
        with torch.no_grad():
            for x, t in dataloader:
                x = x.to(self.device)
                z = self.a_encoder(x)
                zs.append(z.detach().cpu().numpy())
                ts.append(t.detach().cpu().numpy())
                xs.append(x.detach().cpu().numpy())
        zs = np.concatenate(zs, axis=0)
        ts = np.concatenate(ts, axis=0)
        xs = np.concatenate(xs, axis=0)
        xs = xs.reshape(xs.shape[0], -1)
        preds = kmeans.fit_predict(zs)
        cluster_metrics = compute_metrics(ts, preds, xs)

        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)
        cluster_centers = cluster_centers.to(self.device)
        self.state_dict()["clust_layer.cluster_centers"].copy_(cluster_centers)

        self.to(self.device)

        return cluster_metrics
    
    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        ret = (weight.t() / weight.sum(1)).t()

        return ret

    def forward(self, x):
        return self.model(x)


class VADE(nn.Module):
    def __init__(self, latent_dim, n_classes, channel_dims, output_shape, device=None, gmm_n_init=1):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.device = device
        self.gmm_n_init = gmm_n_init
        decoder_channel_dims = channel_dims[::-1]
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.pi = Parameter(torch.FloatTensor(self.n_classes,).fill_(1) / self.n_classes, requires_grad=True)
        self.mu = Parameter(torch.FloatTensor(self.n_classes, self.latent_dim).fill_(0), requires_grad=True)
        self.logvar = Parameter(torch.FloatTensor(self.n_classes, self.latent_dim).fill_(0), requires_grad=True)

        self.encoder = EncoderBlock(channel_dims)
        self.fc_mu = nn.Linear(channel_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(channel_dims[-1], self.latent_dim)
        self.decoder_latent = nn.Linear(self.latent_dim, decoder_channel_dims[0])
        self.decoder = DecoderBlock(decoder_channel_dims, output_shape)

    @property
    def weights(self):
        return F.softmax(self.pi, dim=0)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
    
    def decode(self, z):
        x = self.decoder_latent(z)
        x = self.decoder(x)

        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparametrize(mu, logvar)
        x = self.decode(z)

        return x, mu, logvar
    
    def pretrain(self, dataloader, n_epochs=10, lr=1e-3, add_noise=False, umap_mixture_init=False, grad_clip=None):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(n_epochs):
            for x, _ in dataloader:
                x = x.to(self.device)
                x_train = x.clone()
                if add_noise:
                    x_train = add_gaussian_noise(x_train, 0.1)
                optimizer.zero_grad()
                x_train = x_train.to(self.device)
                z, _ = self.encode(x_train)
                recon_x = self.decode(z)
                loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()

            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}')

        self.eval()
        zs, ts, xs = [], [], []
        for x, t in dataloader:
            x = x.to(self.device)
            z, _ = self.encode(x)
            zs.append(z)
            ts.append(t)
            xs.append(x)

        zs = torch.cat(zs, dim=0).cpu().detach().numpy()
        ts = torch.cat(ts, dim=0).cpu().detach().numpy()
        xs = torch.cat(xs, dim=0).cpu().detach().numpy()
        xs = xs.reshape(xs.shape[0], -1)
        zs = zs.reshape(zs.shape[0], -1)

        if umap_mixture_init:
            umap = UMAP(n_neighbors=100, min_dist=1.0, n_components=self.latent_dim, verbose=True)
            x_transform = umap.fit_transform(xs)
            clusterer = HDBSCAN_flat(x_transform, min_cluster_size=10, cluster_selection_method='eom', n_clusters=self.n_classes)
            clusters = clusterer.labels_
            cluster_metrics = compute_metrics(ts, clusters, zs)
            print(f'Accuracy: {cluster_metrics["acc"]:.4f} | Silhouette: {cluster_metrics["silhouette"]:.4f} | C-H: {cluster_metrics["calinski_harabasz"]:.4f}')

            mixture_weights = np.zeros(self.n_classes)
            mixture_means = np.zeros((self.n_classes, self.latent_dim))
            mixture_logvars = np.zeros((self.n_classes, self.latent_dim))
            for i in range(self.n_classes):
                mixture_weights[i] = np.sum(clusters == i) / len(clusters)
                mixture_means[i] = np.mean(zs[clusters == i], axis=0)
                mixture_logvars[i] = np.log(np.var(zs[clusters == i], axis=0))
            self.pi.data = torch.from_numpy(mixture_weights).float()
            self.mu.data = torch.from_numpy(mixture_means).float()
            self.logvar.data = torch.from_numpy(mixture_logvars).float()

        else:
            while True:
                try:
                    gmm = GaussianMixture(n_components=self.n_classes, covariance_type='diag', n_init=self.gmm_n_init, max_iter=10000)
                    preds = gmm.fit_predict(zs)
                except ValueError:
                    continue
                break
            cluster_metrics = compute_metrics(ts, preds, zs)
            print(f'Accuracy: {cluster_metrics["acc"]:.4f} | Silhouette: {cluster_metrics["silhouette"]:.4f} | C-H: {cluster_metrics["calinski_harabasz"]:.4f}')

            self.pi.data = torch.from_numpy(gmm.weights_).float()
            self.mu.data = torch.from_numpy(gmm.means_).float()
            self.logvar.data = torch.log(torch.from_numpy(gmm.covariances_)).float()

        self.to(self.device)

        return cluster_metrics

    def classify(self, x, n_samples=10):
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = torch.stack([reparametrize(mu, logvar) for _ in range(n_samples)], dim=1)
            log_p_z_given_c = (torch.sum((-0.5 * self.logvar), dim=1) - 
                               0.5 * (math.log(2 * math.pi) + torch.sum(torch.pow((z.unsqueeze(2) - self.mu), 2) / (torch.exp(self.logvar) + 1e-9), dim=3)))
            p_z_c = torch.exp(log_p_z_given_c) * self.weights
            y = p_z_c / (torch.sum(p_z_c, dim=2, keepdim=True) + 1e-9)
            y = torch.sum(y, dim=1)
            pred = torch.argmax(y, dim=1)

        return pred
