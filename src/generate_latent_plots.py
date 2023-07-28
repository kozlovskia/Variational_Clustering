from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader
from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.flat import HDBSCAN_flat

from models import VADE, VAE
from datasets import CustomMNIST, CustomFMNIST, CustomCIFAR10, Brach3, WineQuality, Banknote
from common_utils import reparametrize


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--weights_path', type=Path, default=None)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--dataset_dim', type=int, default=784)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--channel_dims', type=int, nargs='+', default=[784, 500, 500, 2000])
    parser.add_argument('--output_shape', type=int, nargs='+', default=[1, 28, 28])
    parser.add_argument('--perplexity', type=int, default=30)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='vade')
    parser.add_argument('--alpha', type=float, default=0.25)

    return parser.parse_args()


def main(args):
    if args.dataset == 'mnist':
        dataset = CustomMNIST('./data')
    elif args.dataset == 'fmnist':
        dataset = CustomFMNIST('./data')
    elif args.dataset == 'cifar10':
        dataset = CustomCIFAR10('./data')
    elif args.dataset == 'brach3':
        dataset = Brach3('./data/brach3-5klas.txt')
    elif args.dataset == 'winequality':
        dataset = WineQuality('./data/winequality-white.csv')
    elif args.dataset == 'banknote':
        dataset = Banknote('./data/data_banknote_authentication.txt')
    
    if args.model in ('vade', 'vae', 'dgg', 'vade_end_to_end'):
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        if args.model in ('vade', 'vade_end_to_end'):
            model = VADE(args.latent_dim, args.n_clusters, args.channel_dims, args.output_shape, device=args.device)
            model.load_state_dict(torch.load(args.weights_path))
            model.eval()
            model.to(args.device)

            ys = []
            ts = []
            zs = []
            for x, t in loader:
                x = x.to(args.device)
                y = model.classify(x).cpu().detach().numpy()
                z, _ = model.encode(x)
                ys.append(y)
                ts.append(t.cpu().detach().numpy())
                zs.append(z.cpu().detach().numpy())

            ys = np.concatenate(ys, axis=0)
            ts = np.concatenate(ts, axis=0)
            zs = np.concatenate(zs, axis=0)
            zs = zs.reshape(zs.shape[0], -1)
        
        elif args.model == 'vae':
            model = VAE(args.latent_dim, args.channel_dims, args.output_shape)
            model.load_state_dict(torch.load(args.weights_path))
            model.to(args.device)
            model.eval()

            zs, ts = [], []
            for x, t in loader:
                x = x.to(args.device)
                mu, logvar = model.encode(x)
                z = reparametrize(mu, logvar).cpu().detach().numpy()

                zs.append(z)
                ts.append(t.cpu().detach().numpy())
            
            zs = np.concatenate(zs, axis=0)
            ts = np.concatenate(ts, axis=0)
            gmm = GaussianMixture(n_components=args.n_clusters)
            ys = gmm.fit_predict(zs)

    elif args.model.startswith('umap'):
        x = torch.cat([d for d, _ in dataset]).view(-1, args.dataset_dim).numpy()
        ts = np.array([t for _, t in dataset])
        umap = UMAP(n_neighbors=15, min_dist=0.0, n_components=args.latent_dim, verbose=True)
        x_transform = umap.fit_transform(x)

        if args.model == 'umap_gmm':
            clusterer = GaussianMixture(n_components=args.n_clusters, max_iter=10000, n_init=1, verbose=10)
            ys = clusterer.fit_predict(x_transform)
        else:
            clusterer = HDBSCAN_flat(x_transform, min_cluster_size=2, cluster_selection_method='eom', n_clusters=args.n_clusters)
            ys = clusterer.labels_
        
        zs = x_transform

    print('compute silhouette scores')
    silh_samples = silhouette_samples(zs, ys)
    print('done')

    silh_df = pd.DataFrame({'silh_samples': silh_samples, 'y': ys})
    silh_df.y = silh_df.y.astype(int).astype(str)
    silh_df.to_csv(args.save_dir / 'silh_samples.csv', index=False)

    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    sns.violinplot(y='silh_samples', x='y', data=silh_df, palette=sns.color_palette("Set3", 10), order=[str(i) for i in range(args.n_clusters)], ax=ax)
    ax.set(title=f'Silhouette samples for {args.model} latent space', ylabel='silhouette score', xlabel='predicted label')
    plt.tight_layout()
    plt.savefig(args.save_dir / 'silh_samples.png')

    print('fit tsne')
    tsne = TSNE(n_components=2, perplexity=args.perplexity)
    zs_tsne = tsne.fit_transform(zs)
    print('done', zs_tsne.shape)

    if args.latent_dim == 2:
        zs_tsne = zs

    plot_df = pd.DataFrame({'z_1': zs_tsne[:, 0], 'z_2': zs_tsne[:, 1], 'y': ys, 't': ts})
    plot_df[['y', 't']].astype(int).astype(str)
    plot_df.to_csv(args.save_dir / 'plot_df.csv', index=False)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    fig.suptitle(f'TSNE representation of {args.model} latent space')
    sns.scatterplot(x='z_1', y='z_2', hue='y', data=plot_df, ax=axs[0], alpha=args.alpha, palette=sns.color_palette("Set3", args.n_clusters), legend=False)
    axs[0].set(title='predicted labels', xlabel='z_1', ylabel='z_2', xticklabels=[], yticklabels=[])
    sns.scatterplot(x='z_1', y='z_2', hue='t', data=plot_df, ax=axs[1], alpha=args.alpha, palette=sns.color_palette("Set3", args.n_clusters), legend=False)
    axs[1].set(title='true labels', xlabel='z_1', ylabel='z_2', xticklabels=[], yticklabels=[])
    plt.tight_layout()
    plt.savefig(args.save_dir / 'tsne.png')
        

if __name__ == '__main__':
    main(parse_args())
