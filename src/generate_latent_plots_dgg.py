from argparse import ArgumentParser
from pathlib import Path

from DGG.src.generate_latent_plots import main


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--read_dir', type=Path)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=0.25)

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
