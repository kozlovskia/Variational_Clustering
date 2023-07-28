from argparse import ArgumentParser
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from models import VADE
from datasets import CustomMNIST, CustomFMNIST, CustomCIFAR10, Brach3, WineQuality, Banknote
from common_utils import cluster_accuracy, lossfun, add_gaussian_noise, kl_weight_schedule, compute_metrics


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
    parser.add_argument('--pretrain_lr', type=float, default=1e-4)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--gradient_clip', type=float, default=0.2)
    parser.add_argument('--kl_w', type=float, default=1.0)

    parser.add_argument('--kl_div_weight', type=float, default=1.0)
    parser.add_argument('--stabilize', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--wanted_classes', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--add_gaussian_noise', type=bool, default=True)
    parser.add_argument('--gmm_n_init', type=int, default=50)

    parser.add_argument('--better_pretrain', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--gamma_steps', type=int, default=200)
    parser.add_argument('--beta_steps', type=int, default=200)
    parser.add_argument('--tune_steps', type=int, default=300)

    parser.add_argument('--umap_mixture_init', type=bool, default=False)

    parser.add_argument('--description', type=str, default='')
    return parser.parse_args()


def main(args):
    metrics = {'loss_parts_ratios': [], 'loss_history': [], 'accuracy_history': [], 'pretrain_acc_history': [],
               'reconstruction_loss_history': [], 'kl_div_history': [], 'silhouette_score': [], 'calinski_harabasz_score': []}
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

        model = VADE(args.latent_dim, args.n_clusters, args.channel_dims, args.output_shape, device=args.device, gmm_n_init=args.gmm_n_init)
        model.to(args.device)
        if not args.better_pretrain:
            cluster_metrics = model.pretrain(train_loader, n_epochs=args.pretrain_epochs, lr=args.pretrain_lr, 
                                             add_noise=args.add_gaussian_noise, umap_mixture_init=args.umap_mixture_init,
                                             grad_clip=args.gradient_clip)
            torch.save(model.state_dict(), run_dir / f'model_pretrained_{ith_run}.pt')
            metrics['pretrain_acc_history'].append({'run_idx': ith_run, 'pretrain_acc': cluster_metrics['acc']})
            metrics['silhouette_score'].append({'run_idx': ith_run, 'after_pretrain': cluster_metrics['silhouette']})
            metrics['calinski_harabasz_score'].append({'run_idx': ith_run, 'after_pretrain': cluster_metrics['calinski_harabasz']})
            model.to(args.device)
 
        print(model)

        kl_w_schedule = None if not args.better_pretrain else kl_weight_schedule(args.gamma, args.gamma_steps, args.beta_steps, args.tune_steps)
        num_epochs = args.epochs if not args.better_pretrain else args.gamma_steps + args.beta_steps + args.tune_steps
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 50, gamma=0.95)

        for epoch in range(num_epochs):
            kl_div_weight = 0
            if args.better_pretrain:
                kl_div_weight = kl_w_schedule[epoch] * args.kl_w
            else:
                kl_div_weight = args.kl_div_weight * args.kl_w

            model.train()
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            for x, _ in train_loader:
                x = x.to(args.device)
                x_train = x.clone()
                if args.add_gaussian_noise:
                    x_train = add_gaussian_noise(x_train, 0.1)
                    if args.dataset_distr == 'bernoulli':
                        x_train = torch.clamp(x_train, min=1e-9, max=1-1e-9)
                x_train = x_train.to(args.device)

                optimizer.zero_grad()

                recon_x, mu, logvar = model(x_train)
                loss, recon_loss, kl_loss = lossfun(model, x, recon_x, mu, logvar, distr=args.dataset_distr, kl_div_weight=kl_div_weight)
                recon_loss = recon_loss.detach().cpu().item()
                kl_loss = kl_loss.detach().cpu().item()
                
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()

                total_recon_loss += recon_loss
                total_kl_loss += kl_loss

            print(f'Epoch: {epoch}, Loss: {total_loss / len(train_loader):.4f}', end=', ')
            print(f'Recon Loss: {total_recon_loss / len(train_loader):.4f}', end=', ')
            print(f'KL Loss: {total_kl_loss / len(train_loader):.4f}', end=', ')
            if args.better_pretrain and epoch == args.gamma_steps - 1:
                cluster_metrics = model.pretrain(train_loader, n_epochs=0, lr=args.pretrain_lr, add_noise=args.add_gaussian_noise, 
                                                 umap_mixture_init=args.umap_mixture_init, grad_clip=args.gradient_clip)
                torch.save(model.state_dict(), run_dir / f'model_pretrained_{ith_run}.pt')
                metrics['pretrain_acc_history'].append({'run_idx': ith_run, 'pretrain_acc': cluster_metrics['acc']})
                metrics['silhouette_score'].append({'run_idx': ith_run, 'after_pretrain': cluster_metrics['silhouette']})
                metrics['calinski_harabasz_score'].append({'run_idx': ith_run, 'after_pretrain': cluster_metrics['calinski_harabasz']})
                model.to(args.device)
                
            model.eval()
            zs, ys, ts = [], [], []
            for x, t in train_loader:
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
            if epoch == num_epochs - 1:
                cluster_metrics = compute_metrics(ts, ys, zs)
            else:
                cluster_metrics = compute_metrics(ts, ys)
            print(f'Accuracy: {cluster_metrics["acc"]:.4f}')

            lr_scheduler.step()

            metrics['loss_parts_ratios'].append({'elbo_loss_ratio': total_kl_loss / total_recon_loss, 'run_idx': ith_run, 'epoch': epoch})
            metrics['loss_history'].append({'loss': total_loss / len(train_loader), 'run_idx': ith_run, 'epoch': epoch})
            metrics['reconstruction_loss_history'].append({'reconstruction_loss': total_recon_loss / len(train_loader), 'run_idx': ith_run, 'epoch': epoch})
            metrics['kl_div_history'].append({'kl_div': total_kl_loss / len(train_loader), 'run_idx': ith_run, 'epoch': epoch})
            metrics['accuracy_history'].append({'accuracy': cluster_metrics["acc"], 'run_idx': ith_run, 'epoch': epoch})
            if epoch == num_epochs - 1:
                metrics['silhouette_score'][ith_run]['after_train'] = cluster_metrics["silhouette"]
                metrics['calinski_harabasz_score'][ith_run]['after_train'] = cluster_metrics["calinski_harabasz"]

            # tolerance stopping
            if epoch <= args.gamma_steps:
                last_preds = ys.copy()
            else:
                if 1 - cluster_accuracy(last_preds, ys) < args.tolerance:
                    break
                else:
                    last_preds = ys.copy()
            
        torch.save(model.state_dict(), run_dir / f'model_final_{ith_run}.pt')
        
    # save results

    pretrain_acc_history_df = pd.DataFrame(metrics['pretrain_acc_history'])
    pretrain_acc_history_df.run_idx = pretrain_acc_history_df.run_idx.astype(str)
    pretrain_acc_history_df.to_csv(run_dir / 'pretrain_acc_history.csv', index=False)
    fig = plt.figure()
    fig.suptitle('Pretrain Accuracy', fontsize=16)
    sns.barplot(data=pretrain_acc_history_df, x='run_idx', y='pretrain_acc', alpha=0.5)
    plt.savefig(run_dir / 'pretrain_acc.png')

    silhouette_score_df = pd.DataFrame(metrics['silhouette_score'])
    silhouette_score_df.run_idx = silhouette_score_df.run_idx.astype(str)
    silhouette_score_df.to_csv(run_dir / 'silhouette_score.csv', index=False)
    fig = plt.figure()
    fig.suptitle('Silhouette Score', fontsize=16)
    sns.scatterplot(data=silhouette_score_df, x='after_pretrain', y='after_train', alpha=0.9, hue='run_idx')
    plt.savefig(run_dir / 'silhouette_score.png')

    calinski_harabasz_score_df = pd.DataFrame(metrics['calinski_harabasz_score'])
    calinski_harabasz_score_df.run_idx = calinski_harabasz_score_df.run_idx.astype(str)
    calinski_harabasz_score_df.to_csv(run_dir / 'calinski_harabasz_score.csv', index=False)
    fig = plt.figure()
    fig.suptitle('Calinski Harabasz Score', fontsize=16)
    sns.scatterplot(data=calinski_harabasz_score_df, x='after_pretrain', y='after_train', alpha=0.9, hue='run_idx')
    plt.savefig(run_dir / 'calinski_harabasz_score.png')

    lineplots_save_cfg = [('loss_parts_ratios', 'ELBO Loss Ratio: KL Loss / Reconstruction Loss', 'elbo_loss_ratio'),
                          ('loss_history', 'ELBO Loss', 'loss'),
                          ('reconstruction_loss_history', 'Reconstruction Loss', 'reconstruction_loss'),
                          ('kl_div_history', 'KL Divergence', 'kl_div'),
                          ('accuracy_history', 'Accuracy', 'accuracy'),
                          ]
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
