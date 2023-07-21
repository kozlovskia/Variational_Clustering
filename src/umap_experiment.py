from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import torch
from hdbscan import HDBSCAN
from hdbscan.flat import HDBSCAN_flat
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from datasets import CustomMNIST, CustomFMNIST, CustomCIFAR10, Brach3
from common_utils import compute_metrics


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--run_n_times', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--dataset_dim', type=int, default=784)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--n_neighbors', type=int, default=15)
    parser.add_argument('--min_dist', type=float, default=0.)
    parser.add_argument('--cluster_algo', type=str, default='hdbscan')

    parser.add_argument('--min_cluster_size', type=int, default=10)
    parser.add_argument('--description', type=str, default='')

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

    x = torch.cat([d for d, _ in dataset]).view(-1, args.dataset_dim).numpy()
    true_labels = np.array([t for _, t in dataset])

    metrics = {'accuracy': [], 'silhouette_score': [], 'calinski_harabasz_score': []}

    for i in range(args.run_n_times):
        print(f'Run {i + 1}/{args.run_n_times}')
        umap = UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, n_components=args.latent_dim, verbose=True)
        x_transform = umap.fit_transform(x)

        if args.cluster_algo == 'hdbscan':
            clusterer = HDBSCAN_flat(x_transform, min_cluster_size=args.min_cluster_size, cluster_selection_method='eom', n_clusters=args.n_clusters)
            clusterer.fit(x_transform)
            clusters = clusterer.labels_    
        elif args.cluster_algo == 'gmm':
            clusterer = GaussianMixture(n_components=args.n_clusters, max_iter=10000, n_init=50)
            clusters = clusterer.fit_predict(x_transform)

        run_metrics = compute_metrics(true_labels, clusters, x_transform)
        metrics['accuracy'].append(run_metrics['acc'])
        metrics['silhouette_score'].append(run_metrics['silhouette'])
        metrics['calinski_harabasz_score'].append(run_metrics['calinski_harabasz'])

    output_dir = Path("./exp")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not len(list(output_dir.iterdir())):
        run_dir_idx = 0
    else:
        run_dir_idx = max([int(path.stem.split('_')[1]) for path in output_dir.iterdir() if path.is_dir()]) + 1
    run_dir = output_dir / f'run_{run_dir_idx}'
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics)
    metrics_df['run_idx'] = list(range(args.run_n_times))
    print(metrics_df)
    metrics_df.to_csv(run_dir / 'metrics.csv', index=False)

    with open(run_dir / 'args.txt', 'w') as f:
        json.dump(vars(args), f, indent=4)



    # umap_vis = UMAP(n_neighbors=10, min_dist=0., n_components=2)
    # x_transform = umap_vis.fit_transform(x)

    # fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    # cmap = plt.get_cmap("tab10")
    # plot_df = pd.DataFrame(data={'umap_1': x_transform[:, 0], 'umap_2': x_transform[:, 1], 'cluster': clusters.astype(str), 'true_label': true_labels.astype(str)})
    # axs[0].set_title('UMAP ground truth')
    # axs[1].set_title(f'UMAP(8_comp) HDBSCAN cluster | acc: {acc:.4f}')
    # sns.scatterplot(data=plot_df, x='umap_1', y='umap_2', hue='true_label', ax=axs[0], alpha=0.25)
    # sns.scatterplot(data=plot_df, x='umap_1', y='umap_2', hue='cluster', ax=axs[1], alpha=0.25)
    # plt.savefig('umap_experiments/plot.png')


if __name__ == '__main__':
    main(parse_args())
