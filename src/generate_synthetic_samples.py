from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import torch

from models import VADE, VAE
from datasets import CustomMNIST, CustomFMNIST, CustomCIFAR10
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

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='vade')

    return parser.parse_args()


def main(args):
    if args.model in ('vade_e2e', 'vade'):
        print('Loading model...')
        model = VADE(args.latent_dim, args.n_clusters, args.channel_dims, args.output_shape, device=args.device)
        model.load_state_dict(torch.load(args.weights_path))
        model.eval()
        model.to(args.device)

        fig_dim = 32 if args.dataset == 'cifar10' else 28
        fig, axs = plt.subplots(10, 10, figsize=(fig_dim, fig_dim))

        for i in range(args.n_clusters):
            mean = model.mu.data[i]
            logvar = model.logvar.data[i]
            gen_samples = [reparametrize(mean, logvar) for _ in range(10)]
            gen_samples = torch.cat(gen_samples, dim=0).reshape(-1, args.latent_dim)
            gen_samples = model.decode(gen_samples).view(-1, *args.output_shape)
            if args.model == 'vade_e2e':
                gen_samples = torch.sigmoid(gen_samples).data.cpu().numpy()
            else:
                gen_samples = gen_samples.data.cpu().numpy()
            for j in range(10):
                if args.dataset != 'cifar10':
                    axs[i, j].imshow(gen_samples[j][0], cmap='gray')
                else:
                    axs[i, j].imshow(np.transpose(gen_samples[j], (1, 2, 0)))
                axs[i, j].axis('off')

        plt.tight_layout()
        plt.savefig(args.save_dir / 'generated_samples.png')
    
    elif args.model == 'gmm':
        print('Loading dataset...')
        if args.dataset == 'mnist':
            dataset = CustomMNIST('./data')
        elif args.dataset == 'fmnist':
            dataset = CustomFMNIST('./data')
        elif args.dataset == 'cifar10':
            dataset = CustomCIFAR10('./data')

        x = torch.cat([d for d, _ in dataset]).view(-1, args.dataset_dim).numpy()
        gmm = GaussianMixture(n_components=args.n_clusters, covariance_type='diag', max_iter=1000, verbose=10)
        gmm.fit(x)

        means = torch.from_numpy(gmm.means_)
        logvars = torch.from_numpy(gmm.covariances_).log()

        fig_dim = 32 if args.dataset == 'cifar10' else 28
        fig, axs = plt.subplots(10, 10, figsize=(fig_dim, fig_dim))

        for i in range(args.n_clusters):
            mean = means[i]
            logvar = logvars[i]
            gen_samples = [reparametrize(mean, logvar) for _ in range(10)]
            gen_samples = torch.cat(gen_samples, dim=0).reshape(-1, args.dataset_dim)
            gen_samples = gen_samples.view(-1, *args.output_shape).cpu().numpy()
            for j in range(10):
                if args.dataset != 'cifar10':
                    axs[i, j].imshow(gen_samples[j][0], cmap='gray')
                else:
                    axs[i, j].imshow(np.transpose(gen_samples[j], (1, 2, 0)))
                axs[i, j].axis('off')

        plt.tight_layout()
        plt.savefig(args.save_dir / 'generated_samples.png')


if __name__ == '__main__':
    main(parse_args())
