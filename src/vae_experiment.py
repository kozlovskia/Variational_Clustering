from argparse import ArgumentParser
from pathlib import Path
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.mixture import GaussianMixture 
from sklearn.manifold import TSNE

from models import VAE
from datasets import CustomMNIST, CustomFMNIST, CustomCIFAR10, Brach3, WineQuality, Banknote
from common_utils import cluster_accuracy, add_gaussian_noise, vae_loss, reparametrize, compute_metrics


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--run_n_times', type=int, default=1)

    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--dataset_distr', type=str, default='bernoulli')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--channel_dims', type=int, nargs='+', default=[784, 500, 500, 2000])
    parser.add_argument('--output_shape', type=int, nargs='+', default=[1, 28, 28])
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--tolerance', type=float, default=1e-3)

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--wanted_classes', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--add_gaussian_noise', type=bool, default=False)

    parser.add_argument('--description', type=str, default='')
    return parser.parse_args()


def main(args):
    metrics = {'loss_history': [], 'accuracy_history': [],
               'silhouette_score': [], 'calinski_harabasz_score': []}
    output_dir = Path("./exp")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not len(list(output_dir.iterdir())):
        run_dir_idx = 0
    else:
        run_dir_idx = max([int(path.stem.split('_')[1]) for path in output_dir.iterdir() if path.is_dir()]) + 1
    run_dir = output_dir / f'run_{run_dir_idx}'
    run_dir.mkdir(parents=True, exist_ok=True)
    for ith_run in range(args.run_n_times):
        print(f'Run {ith_run + 1}/{args.run_n_times}')
        if args.dataset == 'mnist':
            dataset = CustomMNIST('./data', wanted_classes=args.wanted_classes)
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

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = VAE(args.latent_dim, args.channel_dims, args.output_shape)
        model.to(args.device)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 50, gamma=0.95)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for x, _ in train_loader:
                x = x.to(args.device)
                x_train = x.clone()
                if args.add_gaussian_noise:
                    x_train = add_gaussian_noise(x_train, 0.1)
                    if args.dataset == 'mnist':
                        x_train = torch.clamp(x_train, min=0, max=1)
                x_train = x_train.to(args.device)

                optimizer.zero_grad()
                recon_x, mu, logvar = model(x_train)
                loss = vae_loss(recon_x, x, mu, logvar, distribution=args.dataset_distr)
                loss.backward()
                
                total_loss += loss.detach().cpu().numpy().item()
                optimizer.step()

            print(f'Epoch: {epoch}, Loss: {total_loss / len(train_loader):.4f}')
                
            model.eval()
            zs, ts = [], []
            for x, t in train_loader:
                x = x.to(args.device)
                mu, logvar = model.encode(x)
                z = reparametrize(mu, logvar).cpu().detach().numpy()

                zs.append(z)
                ts.append(t.cpu().detach().numpy())

            if epoch == args.epochs - 1:
                zs = np.concatenate(zs, axis=0)
                ts = np.concatenate(ts, axis=0)
                gmm = GaussianMixture(n_components=args.n_clusters, covariance_type='diag')
                ys = gmm.fit_predict(zs)
                cluster_metrics = compute_metrics(ts, ys, zs)
                print(cluster_metrics)
            else:
                cluster_metrics = {'acc': 0}
   
            # lr_scheduler.step()

            metrics['loss_history'].append({'loss': total_loss / len(train_loader), 'run_idx': ith_run, 'epoch': epoch})
            metrics['accuracy_history'].append({'accuracy': cluster_metrics["acc"], 'run_idx': ith_run, 'epoch': epoch})
            if epoch == args.epochs - 1:
                metrics['silhouette_score'].append({'run_idx': ith_run, 'silhouette': cluster_metrics["silhouette"]})
                metrics['calinski_harabasz_score'].append({'run_idx': ith_run, 'calinski_harabasz': cluster_metrics["calinski_harabasz"]})
        
        tsne = TSNE(n_components=2)
        zs_tsne = tsne.fit_transform(zs)
        if args.latent_dim == 2:
            zs_tsne = zs
        latent_df = pd.DataFrame({'z1': zs_tsne[:, 0], 'z2': zs_tsne[:, 1], 'y': ys, 't': ts})
        latent_df.to_csv(run_dir / f'latent_{ith_run}.csv', index=False)

        torch.save(model.state_dict(), run_dir / f'model_final_{ith_run}.pt')

    # save results
    silhouette_score_df = pd.DataFrame(metrics['silhouette_score'])
    silhouette_score_df.run_idx = silhouette_score_df.run_idx.astype(str)
    silhouette_score_df.to_csv(run_dir / 'silhouette_score.csv', index=False)

    calinski_harabasz_score_df = pd.DataFrame(metrics['calinski_harabasz_score'])
    calinski_harabasz_score_df.run_idx = calinski_harabasz_score_df.run_idx.astype(str)
    calinski_harabasz_score_df.to_csv(run_dir / 'calinski_harabasz_score.csv', index=False)
    
    lineplots_save_cfg = [('loss_history', 'ELBO Loss', 'loss'),
                          ('accuracy_history', 'Accuracy', 'accuracy')]
    for metric_name, title, y in lineplots_save_cfg:
        df = pd.DataFrame(metrics[metric_name])
        df.run_idx = df.run_idx.astype(str)
        df.to_csv(run_dir / f'{metric_name}.csv', index=False)
        fig = px.line(data_frame=df, x='epoch', y=y, color='run_idx', title=title)
        fig.write_html(str(run_dir / f'{metric_name}.html'))
    
    with open(run_dir / 'args.txt', 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':
    main(parse_args())
