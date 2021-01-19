import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from prune import Masking, CosineDecay

def print_and_log(text, log_file):
    """
    Instead of just printing the text to the stdout, also append it to the log file.
    """
    print(text)
    log_dir = './logs'
    log_path = os.path.join(log_dir, log_file)
    if os.path.exists(log_path):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    f = open(log_path, append_write)
    f.write(text + '\n')
    f.close()



def get_accuracy(predictions_batch, labels_batch):
    _, predicted = torch.max(predictions_batch.data, 1)
    total = len(labels_batch)
    correct = (predicted == labels_batch).sum()
    accuracy = 100 * correct / total
    return accuracy

def def_tb_writer_sdr(tb_logdir, n_epochs, batch_size, lr, momentum, nesterov, weight_decay, beta, zeta, zeta_drop, sdr_iter):
    now = datetime.datetime.now()
    date_and_time = now.strftime("%Y%m%d_%H%M")

    NAME = f"run-{date_and_time}-n_epochs_{n_epochs}-batch_size_{batch_size}-lr_{lr}-momentum_{momentum}-nesterov_{nesterov}-weight_decay_{weight_decay}-beta_{beta}-zeta_{zeta}-zeta_drop_{zeta_drop}-sdr_iter_{sdr_iter}"
    writer = SummaryWriter(log_dir=tb_logdir + NAME)
    log_file = 'log_' + NAME + '.txt'
    return writer, log_file

def def_tb_writer(tb_logdir, n_epochs, batch_size, lr, momentum, nesterov, weight_decay):
    now = datetime.datetime.now()
    date_and_time = now.strftime("%Y%m%d_%H%M")

    NAME = f"run-{date_and_time}-n_epochs_{n_epochs}-batch_size_{batch_size}-lr_{lr}-momentum_{momentum}-nesterov_{nesterov}-weight_decay_{weight_decay}"
    writer = SummaryWriter(log_dir=tb_logdir + NAME)
    log_file = 'log_' + NAME + '.txt'
    return writer, log_file

def def_tb_writer_rigL(tb_logdir, n_epochs, batch_size, lr, momentum, nesterov, weight_decay, fraction, distribution, alpha, deltaT, growth_mode):
    now = datetime.datetime.now()
    date_and_time = now.strftime("%Y%m%d_%H%M")

    NAME = f"run-{date_and_time}-n_epochs_{n_epochs}-batch_size_{batch_size}-lr_{lr}-momentum_{momentum}-nesterov_{nesterov}-weight_decay_{weight_decay}-sparsity_{1-fraction}-distribution_{distribution}-alpha_{alpha}-deltaT_{deltaT}-growth_mode_{growth_mode}"
    writer = SummaryWriter(log_dir=tb_logdir + NAME)
    log_file = 'log_' + NAME + '.txt'
    return writer, log_file

def def_tb_writer_rigL_and_sdr(tb_logdir, n_epochs, batch_size, lr, momentum, nesterov, weight_decay, fraction, distribution, alpha, deltaT, beta, zeta, zeta_drop, sdr_iter, growth_mode):
    now = datetime.datetime.now()
    date_and_time = now.strftime("%Y%m%d_%H%M")

    NAME = f"run-{date_and_time}-n_epochs_{n_epochs}-batch_size_{batch_size}-lr_{lr}-momentum_{momentum}-nesterov_{nesterov}-weight_decay_{weight_decay}-sparsity_{1-fraction}-distribution_{distribution}-alpha_{alpha}-deltaT_{deltaT}-beta_{beta}-zeta_{zeta}-zeta_drop_{zeta_drop}-sdr_iter_{sdr_iter}-growth_mode_{growth_mode}"
    writer = SummaryWriter(log_dir=tb_logdir + NAME)
    log_file = 'log_' + NAME + '.txt'
    return writer, log_file


def print_acc_and_loss(epoch, num_batches, accuracy, loss, log_file, training_type='validation'):
    print_and_log(f'epoch: {epoch}, {training_type} loss:     {loss/num_batches}', log_file)
    print_and_log(f'epoch: {epoch}, {training_type} accuracy: {accuracy/num_batches} %', log_file)
    if training_type == 'validation':
        print_and_log('', log_file)

def train_resnet(model,
                 n_epochs,
                 lr,
                 train_dataloader,
                 val_dataloader,
                 GPU_index,
                 batch_size,
                 leonhard=False,
                 momentum=0.9,
                 nesterov=True,
                 weight_decay=1e-4,
                 tb_logdir=None):
    """
    Parameters
    ------------

    *** General Training and Loading ***

    model : pytorch model, initialized before the training procedure
    train_dataloader : pytorch training data dataloader instance
    val_dataloader : pytorch validation data dataloader instance
    batch_size : int
        training batch size
    tb_logdir : str
        tensorboard logdir
    GPU_index : int
        CUDA VISIBLE DEVICES = GPU_index
    leonhard : boolean
        set True if using leonhard cluster

    momentum : float
        SGD momentum by default 0.9
    nesterov : boolean
        SGD nesterov by default True
    weight_decay : float
        SGD L2 weight penalization factor
    lr : float
        SGD learning rate
    n_epochs : int
        number of training epochs

    Returns
    ------------
    NoneType
        None
    """

    if not leonhard:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_index)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            nesterov=nesterov,
                            weight_decay=weight_decay)

    # scheduler will decay initial lr by 0.1 after 50% and another time after 75% of training
    milestones = [n_epochs*0.5, n_epochs*0.75]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    # tensorboard logs:
    if tb_logdir is not None:
        writer, log_file = def_tb_writer(tb_logdir, n_epochs, batch_size, lr, momentum, nesterov, weight_decay)
    else:
        log_file = 'log.txt'



    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.
        num_batches = 0
        running_accuracy = 0.

        for batch_index, batch in enumerate(train_dataloader):
            X_train_batch = batch[0].to(device, dtype=torch.float)
            y_train_batch = batch[1].to(device, dtype=torch.long)
            pred_batch = model(X_train_batch)

            loss=criterion(pred_batch, y_train_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches = batch_index+1

            running_accuracy += get_accuracy(pred_batch, y_train_batch)
        scheduler.step()

        print_acc_and_loss(epoch, num_batches, running_accuracy, running_loss, log_file, 'training')
        if tb_logdir is not None:
            writer.add_scalar("training loss", running_loss/num_batches, global_step=epoch)
            writer.add_scalar("training accuracy", running_accuracy/num_batches, global_step=epoch)

        validation_accuracy = 0.
        validation_loss = 0.
        num_batches = 0
        with torch.no_grad():
            model.eval()
            for batch_index, batch in enumerate(val_dataloader):
                X_val_batch = batch[0].to(device, dtype=torch.float)
                y_val_batch = batch[1].to(device, dtype=torch.long)
                pred_batch = model(X_val_batch)
                loss=criterion(pred_batch, y_val_batch)

                validation_loss += loss.item()
                num_batches = batch_index+1

                validation_accuracy += get_accuracy(pred_batch, y_val_batch)

            print_acc_and_loss(epoch, num_batches, validation_accuracy, validation_loss, log_file, 'validation')
            if tb_logdir is not None:
                writer.add_scalar("validation loss", validation_loss/num_batches, global_step=epoch)
                writer.add_scalar("validation accuracy", validation_accuracy/num_batches, global_step=epoch)

def train_resnet_sdr(model,
                    n_epochs,
                    lr,
                    beta,
                    zeta,
                    zeta_drop,
                    train_dataloader,
                    val_dataloader,
                    GPU_index,
                    batch_size,
                    leonhard=False,
                    momentum=0.9,
                    nesterov=True,
                    weight_decay=1e-4,
                    tb_logdir=None,
                    sdr_iter=195,
                    plot_stds=False):
    """
    Parameters
    ------------

    *** General Training and Loading ***

    model : pytorch model, initialized before the training procedure
    train_dataloader : pytorch training data dataloader instance
    val_dataloader : pytorch validation data dataloader instance
    batch_size : int
        training batch size
    tb_logdir : str
        tensorboard logdir
    GPU_index : int
        CUDA VISIBLE DEVICES = GPU_index
    leonhard : boolean
        set True if using leonhard cluster

    momentum : float
        SGD momentum by default 0.9
    nesterov : boolean
        SGD nesterov by default True
    weight_decay : float
        SGD L2 weight penalization factor
    lr : float
        SGD learning rate equals the learning rate of mu variables in SDR
    n_epochs : int
        number of training epochs

    *** SDR Parameters ***

    beta : float
        stds learning rate
    zeta : float
        stds decay factor
    zeta_drop : float
        zeta decay hyperparameter
    plot_stds : boolean
        set True if one wishes to plot SDR stds to tensorboard
    sdr_iter : int
        number of SGD minibatch iterations until the stds will be updated

    Returns
    ------------
    NoneType
        None
    """

    if not leonhard:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_index)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)



    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            nesterov=nesterov,
                            weight_decay=weight_decay)
    # scheduler will decay initial lr by 0.1 after 50% and another time after 75% of training
    # following https://arxiv.org/abs/1808.03578
    milestones = [n_epochs*0.5, n_epochs*0.75]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    # tensorboard logs:
    if tb_logdir is not None:
        writer, log_file = def_tb_writer_sdr(tb_logdir, n_epochs, batch_size, lr, momentum, nesterov, weight_decay, beta, zeta, zeta_drop, sdr_iter)
    else:
        log_file = 'log.txt'


    zeta_init = zeta
    sds = [] # list for the weight standard deviations
    data_swap = [] # list to hold the means

    iter = 0
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.
        num_batches = 0
        running_accuracy = 0.

        for batch_index, batch in enumerate(train_dataloader):

            # ---- init weight standard deviations for first batch in first epoch ----
            if batch_index == 0 and epoch == 0:
                for name, p in model.named_parameters():
                    if 'bn' not in name:
                        r1 = 0.0
                        r2 = np.sqrt(2. / np.product(p.shape)) * 0.5

                        #normal dist
                        res = torch.randn(p.data.shape)
                        mx = torch.max(res)
                        mn = torch.min(res)

                        #shift distribution so it's between r1 and r1 with
                        #mean (r2-r1)/2
                        init = ((r2 - r1) / (mx - mn)).float() * (res - mn)
                        init.to(device, dtype=torch.float)
                        sds.append([name, init])

                # log initial distribution of stds
                if tb_logdir is not None and plot_stds:
                    for name, param in sds:
                        # only log linear weights and first layer conv weights from the 4 resnet blocks
                        if 'linear' in name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer1.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer2.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer3.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer4.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)

            # ---- update the sds for the individual parameters ----
            if not (batch_index==0 and epoch==0) and iter%sdr_iter == 0:
                k = 0
                for name, p in model.named_parameters():
                    if 'bn' not in name:
                        sds[k][1] = zeta * (torch.abs(beta * p.grad) + sds[k][1]).to(device, dtype=torch.float)
                        k += 1

                # log SDR stds into tb
                if tb_logdir is not None and plot_stds:
                    for name, param in sds:
                        # only log linear weights and first layer conv weights from the 4 resnet blocks
                        if 'linear' in name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer1.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer2.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer3.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer4.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)

            # ---- reset swap list and sample new Wij values for forward pass ----
            data_swap = []
            k = 0
            for name, p in model.named_parameters():
                if 'bn' not in name:
                    p.data = p.data.to(device, dtype=torch.float)
                    sds[k][1] = sds[k][1].to(device, dtype=torch.float)
                    data_swap.append(p.data)
                    # if 'bn' in name:
                    #     print('mean: ',p.data)

                    # sample weights Wij and put them into the original parameter slots
                    p.data = torch.distributions.Normal(p.data, sds[k][1]).sample()
                    # if 'bn' in name:
                    #     print('sampled parameter: ',p.data)
                    k += 1


            # forward pass
            X_train_batch = batch[0].to(device, dtype=torch.float)
            y_train_batch = batch[1].to(device, dtype=torch.long)
            pred_batch = model(X_train_batch)

            # ---- replace sampled Wij values with original mu values for gradient/loss calculations
            # data swap holds the mus, while p's in the model.parameters() are still the sampled Wij
            #for p, s in zip(model.parameters(), data_swap):
            #    p.data = s

            k = 0
            for name, p in model.named_parameters():
                if 'bn' not in name:
                    p.data = data_swap[k]
                    k += 1

            # for k, (name, p) in enumerate(model.named_parameters()):
            #     p.data = data_swap[k]

            # no we calculate gradient updates for the mus as if they were normal weights
            loss=criterion(pred_batch, y_train_batch)

            optimizer.zero_grad()
            loss.backward()

            # for more info see https://github.com/noahfl/sdr-densenet-pytorch/blob/master/train.py
            # lines 404 to 442

            optimizer.step()

            running_loss += loss.item()
            num_batches = batch_index+1

            running_accuracy += get_accuracy(pred_batch, y_train_batch)
            iter += 1
        scheduler.step()

        print_acc_and_loss(epoch, num_batches, running_accuracy, running_loss, log_file, 'training')
        if tb_logdir is not None:
            writer.add_scalar("training loss", running_loss/num_batches, global_step=epoch)
            writer.add_scalar("training accuracy", running_accuracy/num_batches, global_step=epoch)

        validation_accuracy = 0.
        validation_loss = 0.
        num_batches = 0
        with torch.no_grad():
            model.eval()
            for batch_index, batch in enumerate(val_dataloader):
                X_val_batch = batch[0].to(device, dtype=torch.float)
                y_val_batch = batch[1].to(device, dtype=torch.long)
                pred_batch = model(X_val_batch)
                loss=criterion(pred_batch, y_val_batch)

                validation_loss += loss.item()
                num_batches = batch_index+1

                validation_accuracy += get_accuracy(pred_batch, y_val_batch)

            print_acc_and_loss(epoch, num_batches, validation_accuracy, validation_loss, log_file, 'validation')
            if tb_logdir is not None:
                writer.add_scalar("validation loss", validation_loss/num_batches, global_step=epoch)
                writer.add_scalar("validation accuracy", validation_accuracy/num_batches, global_step=epoch)

        # update zeta
        print('zeta value: ', zeta)
        if tb_logdir is not None:
            writer.add_scalar("zeta", zeta, global_step=epoch)
        if (epoch + 1) % zeta_drop == 0:
            # parabolic annealing:
            zeta = zeta_init **((epoch + 1) // zeta_drop)

            # exponential annealing:
            #lambda_ = 0.1
            #zeta = zeta_init * np.power(np.e, -(lambda_ * epoch))


def train_resnet_rigL(model,
                    n_epochs,
                    lr,
                    fraction,
                    distribution,
                    deltaT,
                    alpha,
                    train_dataloader,
                    val_dataloader,
                    GPU_index,
                    batch_size,
                    growth_mode='gradient',
                    leonhard=False,
                    momentum=0.9,
                    nesterov=True,
                    weight_decay=1e-4,
                    T_end = None,
                    tb_logdir=None):
    """
    Parameters
    ------------

    *** General Training and Loading ***

    model : pytorch model, initialized before the training procedure
    train_dataloader : pytorch training data dataloader instance
    val_dataloader : pytorch validation data dataloader instance
    batch_size : int
        training batch size
    tb_logdir : str
        tensorboard logdir
    GPU_index : int
        CUDA VISIBLE DEVICES = GPU_index
    leonhard : boolean
        set True if using leonhard cluster

    momentum : float
        SGD momentum by default 0.9
    nesterov : boolean
        SGD nesterov by default True
    weight_decay : float
        SGD L2 weight penalization factor
    lr : float
        SGD learning rate
    n_epochs : int
        number of training epochs

    *** RigL Parameters ***

    fraction : float
        fraction = 1 - sparsity, the fraction of nonzero weights to keep
    distribution : str
        default uniform, equals sparse_init='constant' in prune.py
    deltaT : int
        number of minibatch SGD iterations between RigL algorithm updates
    alpha : float
        initial prune rate, ie. update factor
    T_end : int
        prune rate decay final iteration by default None and T_end = final minibatch iteration
    growth_mode : str
        set growth_mode='gradient' for RigL

    Returns
    ------------
    NoneType
        None
    """
    # T_end = len(train_dataloader) * n_epochs, maximum number of possible iterations, can be set earlier tho!

    # RigL has to use this, otherwise its momentum growth!!
    assert growth_mode == 'gradient'

    if T_end == None:
        T_end = len(train_dataloader)*n_epochs


    if not leonhard:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_index)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            nesterov=nesterov,
                            weight_decay=weight_decay)

    # tensorboard logs:
    if tb_logdir is not None:
        writer, log_file = def_tb_writer_rigL(tb_logdir, n_epochs, batch_size, lr, momentum, nesterov, weight_decay, fraction, distribution, alpha, deltaT, growth_mode)
    else:
        log_file = 'log.txt'

    alpha_decay = CosineDecay(alpha, T_end)
    mask = Masking(optimizer,
                    prune_rate_decay=alpha_decay,
                    prune_rate=alpha,
                    prune_mode='stable_magnitude',
                    growth_mode=growth_mode,
                    redistribution_mode='none',
                    fp16=False,
                    device=device,
                    deltaT=deltaT,
                    tb_writer=writer)
    mask.add_module(model, density=fraction)

    # in https://arxiv.org/pdf/1911.11134.pdf, the authors leave BatchNorm and Bias params dense
    #mask.remove_weight_partial_name(partial_name='bias')
    #mask.remove_type(nn.BatchNorm2d)

    # scheduler will decay initial lr by 0.1 after 50% and another time after 75% of training
    milestones = [n_epochs*0.5, n_epochs*0.75]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    for epoch in range(1,n_epochs+1):
        model.train()
        running_loss = 0.
        num_batches = 0
        running_accuracy = 0.

        for batch_index, batch in enumerate(train_dataloader):
            # print('batch_index :', batch_index)

            X_train_batch = batch[0].to(device, dtype=torch.float)
            y_train_batch = batch[1].to(device, dtype=torch.long)
            pred_batch = model(X_train_batch)

            loss=criterion(pred_batch, y_train_batch)

            optimizer.zero_grad()
            loss.backward()

            # mask.step() has optimizer.step() and alpha_decay.step() in itself
            # so it performs an optimizer step and an alpha decay step !
            mask.step()

            running_loss += loss.item()
            num_batches = batch_index+1

            running_accuracy += get_accuracy(pred_batch, y_train_batch)
        scheduler.step()

        print_acc_and_loss(epoch, num_batches, running_accuracy, running_loss, log_file, 'training')
        if tb_logdir is not None:
            writer.add_scalar("training loss", running_loss/num_batches, global_step=epoch)
            writer.add_scalar("training accuracy", running_accuracy/num_batches, global_step=epoch)

        validation_accuracy = 0.
        validation_loss = 0.
        num_batches = 0
        with torch.no_grad():
            model.eval()
            for batch_index, batch in enumerate(val_dataloader):
                X_val_batch = batch[0].to(device, dtype=torch.float)
                y_val_batch = batch[1].to(device, dtype=torch.long)
                pred_batch = model(X_val_batch)
                loss=criterion(pred_batch, y_val_batch)

                validation_loss += loss.item()
                num_batches = batch_index+1

                validation_accuracy += get_accuracy(pred_batch, y_val_batch)

            print_acc_and_loss(epoch, num_batches, validation_accuracy, validation_loss, log_file, 'validation')
            if tb_logdir is not None:
                writer.add_scalar("validation loss", validation_loss/num_batches, global_step=epoch)
                writer.add_scalar("validation accuracy", validation_accuracy/num_batches, global_step=epoch)

        # after every epoch, the masking is reasessed and weights are pruned and grown
        # look at truncate_weights in prune.py
        if epoch < n_epochs:
            mask.at_end_of_epoch()

    ### Print number of nonzero weights in model ###
    num_zeros = countZeroWeights(model)
    non_zero_weights = countNonzeroWeights(model)

    print('='*50)
    print_and_log(f"Non-Zero Weights: {non_zero_weights}", log_file)
    print_and_log(f"Zero Weights: {num_zeros}", log_file)
    print('='*50)

def countZeroWeights(model):
    zeros = 0
    for param in model.parameters():
        if param is not None:
            try:
                zeros += torch.sum((param == 0).int()).data[0]
            except:
                zeros += torch.sum((param == 0).int()).item()
    return zeros


def countNonzeroWeights(model):
    nonzeros = 0
    for param in model.parameters():
        try:
            nonzeros += torch.sum((param != 0).int()).data[0]
        except:
            nonzeros += torch.sum((param != 0).int()).item()
    return nonzeros


def train_resnet_momgrowth(model,
                    n_epochs,
                    lr,
                    fraction,
                    distribution,
                    deltaT,
                    alpha,
                    train_dataloader,
                    val_dataloader,
                    GPU_index,
                    batch_size,
                    growth_mode='momentum',
                    leonhard=False,
                    momentum=0.9,
                    nesterov=True,
                    weight_decay=1e-4,
                    T_end = None,
                    tb_logdir=None):
    """
    Parameters
    ------------

    *** General Training and Loading ***

    model : pytorch model, initialized before the training procedure
    train_dataloader : pytorch training data dataloader instance
    val_dataloader : pytorch validation data dataloader instance
    batch_size : int
        training batch size
    tb_logdir : str
        tensorboard logdir
    GPU_index : int
        CUDA VISIBLE DEVICES = GPU_index
    leonhard : boolean
        set True if using leonhard cluster

    momentum : float
        SGD momentum by default 0.9
    nesterov : boolean
        SGD nesterov by default True
    weight_decay : float
        SGD L2 weight penalization factor
    lr : float
        SGD learning rate
    n_epochs : int
        number of training epochs

    *** SNFS/Momentum Parameters ***
    SNFS = "sparse networks from scratch" name by Dettmers&Zettlemoyer

    fraction : float
        fraction = 1 - sparsity, the fraction of nonzero weights to keep
    distribution : str
        default uniform, equals sparse_init='constant' in prune.py
    deltaT : int
        set None to update pruning and growth at every end of epoch
    alpha : float
        initial prune rate, ie. update factor
    T_end : int
        prune rate decay final iteration by default None and T_end = final minibatch iteration
    growth_mode : str
        set growth_mode='momentum' for SNFS

    Returns
    ------------
    NoneType
        None
    """
    # T_end = len(train_dataloader) * n_epochs, maximum number of possible iterations, can be set earlier tho!
    assert deltaT is None
    assert growth_mode == 'momentum'

    if T_end == None:
        T_end = len(train_dataloader)*n_epochs


    if not leonhard:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_index)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            nesterov=nesterov,
                            weight_decay=weight_decay)

    # tensorboard logs:
    if tb_logdir is not None:
        writer, log_file = def_tb_writer_rigL(tb_logdir, n_epochs, batch_size, lr, momentum, nesterov, weight_decay, fraction, distribution, alpha, deltaT, growth_mode)
    else:
        log_file = 'log.txt'


    alpha_decay = CosineDecay(alpha, T_end)
    mask = Masking(optimizer,
                    prune_rate_decay=alpha_decay,
                    prune_rate=alpha,
                    prune_mode='magnitude',
                    growth_mode=growth_mode,
                    redistribution_mode='momentum',
                    fp16=False,
                    device=device,
                    deltaT=deltaT,
                    tb_writer=writer)
    mask.add_module(model, density=fraction)

    # in https://arxiv.org/pdf/1911.11134.pdf, the authors leave BatchNorm and Bias params dense
    #mask.remove_weight_partial_name(partial_name='bias')
    #mask.remove_type(nn.BatchNorm2d)

    # scheduler will decay initial lr by 0.1 after 50% and another time after 75% of training
    milestones = [n_epochs*0.5, n_epochs*0.75]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    for epoch in range(1,n_epochs+1):
        model.train()
        running_loss = 0.
        num_batches = 0
        running_accuracy = 0.

        for batch_index, batch in enumerate(train_dataloader):
            # print('batch_index :', batch_index)

            X_train_batch = batch[0].to(device, dtype=torch.float)
            y_train_batch = batch[1].to(device, dtype=torch.long)
            pred_batch = model(X_train_batch)

            loss=criterion(pred_batch, y_train_batch)

            optimizer.zero_grad()
            loss.backward()

            # mask.step() has optimizer.step() and alpha_decay.step() in itself
            # so it performs an optimizer step and an alpha decay step !
            mask.step()

            running_loss += loss.item()
            num_batches = batch_index+1

            running_accuracy += get_accuracy(pred_batch, y_train_batch)
        scheduler.step()

        print_acc_and_loss(epoch, num_batches, running_accuracy, running_loss, log_file, 'training')
        if tb_logdir is not None:
            writer.add_scalar("training loss", running_loss/num_batches, global_step=epoch)
            writer.add_scalar("training accuracy", running_accuracy/num_batches, global_step=epoch)

        validation_accuracy = 0.
        validation_loss = 0.
        num_batches = 0
        with torch.no_grad():
            model.eval()
            for batch_index, batch in enumerate(val_dataloader):
                X_val_batch = batch[0].to(device, dtype=torch.float)
                y_val_batch = batch[1].to(device, dtype=torch.long)
                pred_batch = model(X_val_batch)
                loss=criterion(pred_batch, y_val_batch)

                validation_loss += loss.item()
                num_batches = batch_index+1

                validation_accuracy += get_accuracy(pred_batch, y_val_batch)

            print_acc_and_loss(epoch, num_batches, validation_accuracy, validation_loss, log_file, 'validation')
            if tb_logdir is not None:
                writer.add_scalar("validation loss", validation_loss/num_batches, global_step=epoch)
                writer.add_scalar("validation accuracy", validation_accuracy/num_batches, global_step=epoch)

        # after every epoch, the masking is reasessed and weights are pruned and grown
        # look at truncate_weights in prune.py
        if epoch < n_epochs:
            mask.at_end_of_epoch()



def train_resnet_rigL_and_sdr(model,
                    n_epochs,
                    lr,
                    fraction,
                    distribution,
                    deltaT,
                    alpha,
                    beta,
                    zeta,
                    zeta_drop,
                    train_dataloader,
                    val_dataloader,
                    GPU_index,
                    batch_size,
                    growth_mode='redistributed_gradient',
                    prune_mode='magnitude',
                    redistribution_mode='reverse_std_redistribution',
                    leonhard=False,
                    momentum=0.9,
                    nesterov=True,
                    weight_decay=1e-4,
                    T_end = None,
                    tb_logdir=None,
                    plot_stds=False,
                    sdr_iter=195):
    """
    Parameters
    ------------

    *** General Training and Loading ***

    model : pytorch model, initialized before the training procedure
    train_dataloader : pytorch training data dataloader instance
    val_dataloader : pytorch validation data dataloader instance
    batch_size : int
        training batch size
    tb_logdir : str
        tensorboard logdir
    GPU_index : int
        CUDA VISIBLE DEVICES = GPU_index
    leonhard : boolean
        set True if using leonhard cluster

    momentum : float
        SGD momentum by default 0.9
    nesterov : boolean
        SGD nesterov by default True
    weight_decay : float
        SGD L2 weight penalization factor
    lr : float
        SGD learning rate equals the learning rate of mu variables in SDR
    n_epochs : int
        number of training epochs

    *** RigL Parameters ***

    fraction : float
        fraction = 1 - sparsity, the fraction of nonzero weights to keep
    distribution : str
        default uniform, equals sparse_init='constant' in prune.py
    deltaT : int
        number of minibatch SGD iterations between RigL algorithm updates
    alpha : float
        initial prune rate, ie. update factor
    T_end : int
        prune rate decay final iteration by default None and T_end = final minibatch iteration
    growth_mode : str
        set growth_mode='gradient' for RigL

    *** SDR Parameters ***

    beta : float
        stds learning rate
    zeta : float
        stds decay factor
    zeta_drop : float
        zeta decay hyperparameter
    plot_stds : boolean
        set True if one wishes to plot SDR stds to tensorboard
    sdr_iter : int
        number of SGD minibatch iterations until the stds will be updated

    Returns
    ------------
    NoneType
        None
    """
    # T_end = len(train_dataloader) * n_epochs, maximum number of possible iterations, can be set earlier tho!

    # Required for RigL+SDR
    #assert growth_mode == 'redistributed_gradient'

    if T_end == None:
        T_end = len(train_dataloader)*n_epochs

    if not leonhard:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_index)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            nesterov=nesterov,
                            weight_decay=weight_decay)

    zeta_init = zeta
    sds = [] # list for the weight standard deviations
    data_swap = [] # list to hold the means

    # tensorboard logs:
    if tb_logdir is not None:
        writer, log_file = def_tb_writer_rigL_and_sdr(tb_logdir, n_epochs, batch_size, lr, momentum, nesterov, weight_decay, fraction, distribution, alpha, deltaT, beta, zeta, zeta_drop, sdr_iter, growth_mode)
    else:
        log_file = 'log.txt'

    alpha_decay = CosineDecay(alpha, T_end)
    mask = Masking(optimizer,
                    sds_init = sds,
                    prune_rate_decay=alpha_decay,
                    prune_rate=alpha,
                    prune_mode=prune_mode,
                    growth_mode=growth_mode,
                    redistribution_mode=redistribution_mode,
                    fp16=False,
                    device=device,
                    deltaT=deltaT,
                    tb_writer=writer)
    mask.add_module(model, density=fraction)

    # in https://arxiv.org/pdf/1911.11134.pdf, the authors leave BatchNorm and Bias params dense
    #mask.remove_weight_partial_name(partial_name='bias')
    #mask.remove_type(nn.BatchNorm2d)

    # scheduler will decay initial lr by 0.1 after 50% and another time after 75% of training
    milestones = [n_epochs*0.5, n_epochs*0.75]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    iter = 0
    for epoch in range(1,n_epochs+1):
        model.train()
        running_loss = 0.
        num_batches = 0
        running_accuracy = 0.

        for batch_index, batch in enumerate(train_dataloader):

            # ---- init weight standard deviations for first batch in first epoch ----
            if batch_index == 0 and epoch == 1: # CAREFUL! in RigL the epochs loop starts at 1!!
                for name, p in model.named_parameters():
                    if 'bn' not in name:
                        r1 = 0.0
                        r2 = np.sqrt(2. / np.product(p.shape)) * 0.5

                        #normal dist
                        res = torch.randn(p.data.shape)
                        mx = torch.max(res)
                        mn = torch.min(res)

                        #shift distribution so it's between r1 and r1 with
                        #mean (r2-r1)/2
                        init = ((r2 - r1) / (mx - mn)).float() * (res - mn)
                        init.to(device, dtype=torch.float)
                        sds.append([name, init])

                # log initial distribution of stds
                if tb_logdir is not None and plot_stds:
                    for name, param in sds:
                        # only log linear weights and first layer conv weights from the 4 resnet blocks
                        if 'linear' in name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer1.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer2.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer3.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer4.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)

            # ---- update the sds for the individual parameters ----
            if not (batch_index==0 and epoch==1) and iter%sdr_iter == 0:
                print('updating the stds at iter: ', iter)
                k = 0
                for name, p in model.named_parameters():
                    if 'bn' not in name:
                        # update standard deviations:
                        sds[k][1] = zeta * (torch.abs(beta * p.grad) + sds[k][1]).to(device, dtype=torch.float)
                        k += 1

                # log SDR stds into tb
                if tb_logdir is not None and plot_stds:
                    for name, param in sds:
                        # only log linear weights and first layer conv weights from the 4 resnet blocks
                        if 'linear' in name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer1.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer2.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer3.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)
                        elif 'layer4.0.conv1.weight'==name:
                            writer.add_histogram('SDR-stds_' + name, param.cpu().numpy().ravel(), global_step=epoch)
                            writer.add_scalar('mean_std_' + name, param.mean(), global_step=epoch)

            # ---- reset swap list and sample new Wij values for forward pass ----
            data_swap = []
            k = 0
            for name, p in model.named_parameters():
                if 'bn' not in name:
                    p.data = p.data.to(device, dtype=torch.float)
                    sds[k][1] = sds[k][1].to(device, dtype=torch.float)
                    data_swap.append(p.data)

                    # sample weights Wij and put them into the original parameter slots
                    p.data = torch.distributions.Normal(p.data, sds[k][1]).sample()
                    k += 1

            # apply mask after the SDR sampling step!
            mask.apply_mask()

            # forward pass
            X_train_batch = batch[0].to(device, dtype=torch.float)
            y_train_batch = batch[1].to(device, dtype=torch.long)
            pred_batch = model(X_train_batch)

            # ---- replace sampled Wij values with original mu values for gradient/loss calculations
            # data swap holds the mus, while p's in the model.parameters() are still the sampled Wij
            #for p, s in zip(model.parameters(), data_swap):
            #    p.data = s

            k = 0
            for name, p in model.named_parameters():
                if 'bn' not in name:
                    p.data = data_swap[k]
                    k += 1

            # no we calculate gradient updates for the mus as if they were normal weights
            loss=criterion(pred_batch, y_train_batch)

            optimizer.zero_grad()
            loss.backward()

            # update the sds in the Masking class
            mask.get_sds(sds, zeta)

            # for more info see https://github.com/noahfl/sdr-densenet-pytorch/blob/master/train.py
            # lines 404 to 442

            # mask.step() has optimizer.step() and alpha_decay.step() in itself
            # so it performs an optimizer step and an alpha decay step !
            mask.step()

            running_loss += loss.item()
            num_batches = batch_index+1

            running_accuracy += get_accuracy(pred_batch, y_train_batch)
            iter += 1
        scheduler.step()

        print_acc_and_loss(epoch, num_batches, running_accuracy, running_loss, log_file, 'training')
        if tb_logdir is not None:
            writer.add_scalar("training loss", running_loss/num_batches, global_step=epoch)
            writer.add_scalar("training accuracy", running_accuracy/num_batches, global_step=epoch)

        validation_accuracy = 0.
        validation_loss = 0.
        num_batches = 0
        with torch.no_grad():
            mask.apply_mask()
            model.eval()
            for batch_index, batch in enumerate(val_dataloader):
                X_val_batch = batch[0].to(device, dtype=torch.float)
                y_val_batch = batch[1].to(device, dtype=torch.long)
                pred_batch = model(X_val_batch)
                loss=criterion(pred_batch, y_val_batch)

                validation_loss += loss.item()
                num_batches = batch_index+1

                validation_accuracy += get_accuracy(pred_batch, y_val_batch)

            print_acc_and_loss(epoch, num_batches, validation_accuracy, validation_loss, log_file, 'validation')
            if tb_logdir is not None:
                writer.add_scalar("validation loss", validation_loss/num_batches, global_step=epoch)
                writer.add_scalar("validation accuracy", validation_accuracy/num_batches, global_step=epoch)

        # after every epoch, the masking is reasessed and weights are pruned and grown
        # look at truncate_weights in prune.py
        # update zeta
        print('zeta value: ', zeta)
        if tb_logdir is not None:
            writer.add_scalar("zeta", zeta, global_step=epoch)
        if (epoch + 1) % zeta_drop == 0:
            # parabolic annealing:
            zeta = zeta_init **((epoch + 1) // zeta_drop)

            # exponential annealing:
            #lambda_ = 0.1
            #zeta = zeta_init * np.power(np.e, -(lambda_ * epoch))

        if epoch < n_epochs:
            mask.at_end_of_epoch()
