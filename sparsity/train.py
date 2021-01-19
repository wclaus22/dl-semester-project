import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torch
from models.resnet import ResNet50
from utils import train_resnet_rigL_and_sdr, train_resnet, train_resnet_sdr, train_resnet_rigL, train_resnet_momgrowth
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n_epochs', help='number of training epochs', default=100, type=int)
parser.add_argument('--batch_size', help='training batch size', default=128, type=int)
parser.add_argument('--lr', help='training learning rate', default=0.05, type=float)
parser.add_argument('--weight_decay', help='L2 weight decay', default=1e-3, type=float)
parser.add_argument('--momentum', help='SGD momentum', default=0.9, type=float)

parser.add_argument('--training_type', help='choose (sparse) training method', default='Vanilla', type=str)
parser.add_argument('--fraction', help='fraction of weights to keep - sparsity=0.99 equals fraction=0.01', default=0.01, type=float)

parser.add_argument('--zeta', help='SDR zeta parameter', default=0.9, type=float)
parser.add_argument('--beta', help='SDR beta parameter', default=0.1, type=float)

parser.add_argument('--alpha', help='RigL/SNFS alpha parameter', default=0.1, type=float)
parser.add_argument('--deltaT', help='RigL deltaT parameter', default=100, type=int)

parser.add_argument('--GPU_index', help='choose GPU index for CUDA VISIBLE DEVICES', default=0, type=int)
parser.add_argument('--leonhard', help='set True if training on Leonhard cluster', default=False, type=bool)

parser.add_argument('--seed', help='init. and batch shuffle seed', default=42, type=int)

args = parser.parse_args()

def get_CIFAR10_data(path, batch_size, num_workers):

    augment = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    ])

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)

    normalize = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
    ])

    train_transforms = torchvision.transforms.Compose([
    augment,
    normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR10(path + '/data/CIFAR10', train=True, transform=train_transforms, download=True)
    val_dataset = torchvision.datasets.CIFAR10(path + '/data/CIFAR10', train=False, transform=normalize, download=True)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=256, num_workers=num_workers)
    return train_data_loader, val_data_loader


def get_CIFAR100_data(path, batch_size, num_workers):

    augment = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    ])

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    normalize = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
    ])

    train_transforms = torchvision.transforms.Compose([
    augment,
    normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR100(path + '/data/CIFAR100', train=True, transform=train_transforms, download=True)
    val_dataset = torchvision.datasets.CIFAR100(path + '/data/CIFAR100', train=False, transform=normalize, download=True)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=256, num_workers=num_workers)
    return train_data_loader, val_data_loader

if __name__ == '__main__':
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        os.makedirs('./sparsity')
    except:
        pass

    try:
        os.makedirs('./logs')
    except:
        pass

    try:
        os.makedirs('./tb_logs')
    except:
        pass

    train_dataloader, val_dataloader = get_CIFAR100_data(path='./sparsity', batch_size=args.batch_size, num_workers=4)
    tb_logdir = f'./tb_logs/tensorboard_{args.training_type}_'

    net = ResNet50()

    # len(train_dataloader) == 391 the number of iterations for one epoch ie. the number of batches of size 128
    # sdr_iter = number of minibatch gradient descent iterations until an SDR update step
    sdr_iter = len(train_dataloader)//2
    print('sdr_iter is :', sdr_iter)
    # deactivate zeta annealing with an annealing step at epoch 100, can be activated for hyperbolic zeta annealing
    zeta_drop = 100
    distribution = 'uniform'

    if args.training_type == 'Vanilla':
        # print the number of model parameters
        print(f'Number of model parameters: {sum([p.data.nelement() for p in net.parameters()])/1e6} million params')
        print(f"Training Type: {args.training_type} with ResNet50 for {args.n_epochs} epochs.")
        train_resnet(net,
                     args.n_epochs,
                     args.lr,
                     train_dataloader,
                     val_dataloader,
                     GPU_index=args.GPU_index,
                     batch_size=args.batch_size,
                     momentum=args.momentum,
                     nesterov=True,
                     leonhard=args.leonhard,
                     weight_decay=args.weight_decay,
                     tb_logdir=tb_logdir)

    elif args.training_type == 'SDR':
        # print the number of model parameters
        print(f'Number of model parameters: {sum([p.data.nelement() for p in net.parameters()])/1e6} million params')
        print(f"Training Type: {args.training_type} with ResNet50 for {args.n_epochs} epochs.")
        train_resnet_sdr(net,
                         args.n_epochs,
                         args.lr,
                         args.beta,
                         args.zeta,
                         zeta_drop,
                         train_dataloader,
                         val_dataloader,
                         GPU_index=args.GPU_index,
                         batch_size=args.batch_size,
                         leonhard=args.leonhard,
                         momentum=args.momentum,
                         nesterov=True,
                         weight_decay=args.weight_decay,
                         tb_logdir=tb_logdir,
                         sdr_iter=sdr_iter,
                         plot_stds=True)

    elif args.training_type == 'RigL':
        # print the number of model parameters
        print(f'Number of model parameters: {sum([p.data.nelement() for p in net.parameters()])/1e6} million params')
        print(f"Training Type: {args.training_type} with ResNet50 for {args.n_epochs} epochs.")
        train_resnet_rigL(net,
                          args.n_epochs,
                          args.lr,
                          args.fraction,
                          distribution,
                          args.deltaT,
                          args.alpha,
                          train_dataloader,
                          val_dataloader,
                          args.GPU_index,
                          args.batch_size,
                          growth_mode='gradient',
                          leonhard=args.leonhard,
                          momentum=args.momentum,
                          nesterov=True,
                          weight_decay=args.weight_decay,
                          T_end = None,
                          tb_logdir=tb_logdir)

    elif args.training_type == 'SNFS':
        print(f'Number of model parameters: {sum([p.data.nelement() for p in net.parameters()])/1e6} million params')
        print(f"Training Type: {args.training_type} with ResNet50 for {args.n_epochs} epochs.")
        train_resnet_momgrowth(net,
                           args.n_epochs,
                           args.lr,
                           args.fraction,
                           distribution,
                           args.deltaT,
                           args.alpha,
                           train_dataloader,
                           val_dataloader,
                           args.GPU_index,
                           args.batch_size,
                           growth_mode='momentum',
                           leonhard=args.leonhard,
                           momentum=args.momentum,
                           nesterov=True,
                           weight_decay=args.weight_decay,
                           T_end = None,
                           tb_logdir=tb_logdir)

    elif args.training_type == 'sigma-redistribution':
        print(f'Number of model parameters: {sum([p.data.nelement() for p in net.parameters()])/1e6} million params')
        print(f"Training Type: {args.training_type} with ResNet50 for {args.n_epochs} epochs.")
        train_resnet_rigL_and_sdr(net,
                                  args.n_epochs,
                                  args.lr,
                                  args.fraction,
                                  distribution,
                                  args.deltaT,
                                  args.alpha,
                                  args.beta,
                                  args.zeta,
                                  zeta_drop,
                                  train_dataloader,
                                  val_dataloader,
                                  args.GPU_index,
                                  args.batch_size,
                                  growth_mode='redistributed_gradient',
                                  prune_mode='magnitude',
                                  redistribution_mode='reverse_std_redistribution',
                                  leonhard=args.leonhard,
                                  momentum=args.momentum,
                                  nesterov=True,
                                  weight_decay=args.weight_decay,
                                  T_end = None,
                                  tb_logdir=tb_logdir,
                                  plot_stds=True,
                                  sdr_iter=sdr_iter)

    elif args.training_type == 'sigma-pruning-rigl':
        print(f'Number of model parameters: {sum([p.data.nelement() for p in net.parameters()])/1e6} million params')
        print(f"Training Type: {args.training_type} with ResNet50 for {args.n_epochs} epochs.")
        train_resnet_rigL_and_sdr(net,
                                  args.n_epochs,
                                  args.lr,
                                  args.fraction,
                                  distribution,
                                  args.deltaT,
                                  args.alpha,
                                  args.beta,
                                  args.zeta,
                                  zeta_drop,
                                  train_dataloader,
                                  val_dataloader,
                                  args.GPU_index,
                                  args.batch_size,
                                  growth_mode='gradient',
                                  prune_mode='stable_std_prune',
                                  redistribution_mode='none',
                                  leonhard=args.leonhard,
                                  momentum=args.momentum,
                                  nesterov=True,
                                  weight_decay=args.weight_decay,
                                  T_end = None,
                                  tb_logdir=tb_logdir,
                                  plot_stds=True,
                                  sdr_iter=sdr_iter)

    elif args.training_type == 'sigma-pruning-SNFS':
        print(f'Number of model parameters: {sum([p.data.nelement() for p in net.parameters()])/1e6} million params')
        print(f"Training Type: {args.training_type} with ResNet50 for {args.n_epochs} epochs.")
        train_resnet_rigL_and_sdr(net,
                                  args.n_epochs,
                                  args.lr,
                                  args.fraction,
                                  distribution,
                                  args.deltaT,
                                  args.alpha,
                                  args.beta,
                                  args.zeta,
                                  zeta_drop,
                                  train_dataloader,
                                  val_dataloader,
                                  args.GPU_index,
                                  args.batch_size,
                                  growth_mode='momentum',
                                  prune_mode='std_prune',
                                  redistribution_mode='momentum',
                                  leonhard=args.leonhard,
                                  momentum=args.momentum,
                                  nesterov=True,
                                  weight_decay=args.weight_decay,
                                  T_end = None,
                                  tb_logdir=tb_logdir,
                                  plot_stds=True,
                                  sdr_iter=sdr_iter)
