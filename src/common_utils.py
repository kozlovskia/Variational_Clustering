import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mu + eps * std


def vae_loss(recon_x, x, mu, logvar, distribution='bernoulli', kl_div_weight=1.0):
    kl_div = torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(logvar) - logvar - 1, dim=1), dim=0)
    kl_div *= kl_div_weight
    if distribution == 'bernoulli':     
        recon_loss = F.binary_cross_entropy(torch.sigmoid(recon_x), x, reduction='sum') / x.size(0)    
        return recon_loss + kl_div
    elif distribution == 'normal':
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        return recon_loss + kl_div
    else:
        raise NotImplementedError


def lossfun(model, x, recon_x, mu, logvar, distr='bernoulli', kl_div_weight=1.0):
    batch_size = x.size(0)
    z = reparametrize(mu, logvar).unsqueeze(1)
    h = z - model.mu
    h = torch.exp(-0.5 * torch.sum((h * h / model.logvar.exp()), dim=2))
    h = h / torch.sum(0.5 * model.logvar, dim=1).exp()
    p_z_given_c = h / (2 * math.pi)
    p_z_c = p_z_given_c * (model.weights) + 1e-9
    gamma = p_z_c / (torch.sum(p_z_c, dim=1, keepdim=True))
    h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - model.mu).pow(2)
    h = torch.sum(model.logvar + h / model.logvar.exp(), dim=2)
    if distr == 'bernoulli':
        recon_loss = F.binary_cross_entropy(torch.sigmoid(recon_x), x, reduction='sum')
        rest = 0.5 * torch.sum(gamma * h) \
            - torch.sum(gamma * torch.log(model.weights + 1e-9)) \
            + torch.sum(gamma * torch.log(gamma + 1e-9)) \
            - 0.5 * torch.sum(1 + logvar)
        loss = recon_loss + kl_div_weight * rest
    elif distr == 'normal':
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        rest = 0.5 * torch.sum(gamma * h) \
            - torch.sum(gamma * torch.log(model.weights + 1e-9)) \
            + torch.sum(gamma * torch.log(gamma + 1e-9)) \
            - 0.5 * torch.sum(1 + logvar)
        loss = recon_loss + kl_div_weight * rest
    # print(f"recon_loss: {recon_loss}, rest: {rest}, sum_gamma_h: {torch.sum(gamma * h)}\n{gamma}", end='\n====================\n')
    loss = loss / batch_size
    return loss, recon_loss / batch_size, (kl_div_weight * rest) / batch_size


def cluster_accuracy(y_true, y_pred):
    """ Compute clustering accuracy. """
    y_true = y_true[y_pred >= 0]
    y_pred = y_pred[y_pred >= 0]
    dim = max(y_pred.max(), y_true.max()) + 1
    cost_mat = np.zeros((dim, dim), dtype=np.int64)
    for i in range(len(y_pred)):
        cost_mat[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(cost_mat.max() - cost_mat)

    return sum([cost_mat[i, j] for i, j in zip(*ind)]) * 1.0 / np.sum(cost_mat)


def add_gaussian_noise(x, sigma=0.1):
    """ Add gaussian noise to input tensor. """
    x = x + torch.randn_like(x) * sigma
    return x


def kl_weight_schedule(gamma, t_pretrain, t_train, t_tune):
    w_pretrain = np.repeat(gamma, t_pretrain)
    w_train = np.linspace(0, 1, t_train) + gamma
    w_tune = np.repeat(1, t_tune)
    return np.concatenate((w_pretrain, w_train, w_tune))


def compute_metrics(y_true, y_pred, x=None):
    ret = dict()
    ret['acc'] = cluster_accuracy(y_true, y_pred)
    if x is not None:
        try:
            ret['silhouette'] = silhouette_score(x, y_pred)
        except:
            ret['silhouette'] = -1
        try:
            ret['calinski_harabasz'] = calinski_harabasz_score(x, y_pred)
        except:
            ret['calinski_harabasz'] = -1

    return ret
