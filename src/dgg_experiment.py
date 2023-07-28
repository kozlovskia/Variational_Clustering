from pathlib import Path
from argparse import ArgumentParser
import json

from DGG.src.Siamesev0 import main as siamesev0_main
from DGG.src.Form_data_using_Siamese import main as form_data_using_siamese_main
from DGG.src.DGG import main as dgg_main


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--run_idx', type=int, default=0)
    parser.add_argument('--dataset_dim', type=int, default=784)
    parser.add_argument('--channel_dims', type=int, nargs='+', default=[784, 500, 500, 2000])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--n_epochs', default=300, type=int)
    parser.add_argument('--latent_dim', default=10, type=int)
    parser.add_argument('--dataset_distr', type=str, default='bernoulli')
    parser.add_argument('--num_neighbors', type=int, default=21)

    parser.add_argument('--form_data', action='store_true')

    return parser.parse_args()


def main(args):
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.form_data:
        siamesev0_main(args)
        form_data_using_siamese_main(args)
    for i in range(10):
        args.run_idx = i
        print(f'Run {i + 1}/{10}')
        dgg_main(args)

    with open(args.save_dir / 'args.txt', 'w') as f:
        args.save_dir = str(args.save_dir)
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':
    main(parse_args())
