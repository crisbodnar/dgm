import argparse
import torch
import datetime
import os
import os.path as osp
import random
import numpy as np
import torch.nn.functional as F

from mpr.pmodels import MPRModel, StandardPoolingModel, FlatModel, AverageMLP, GINModel

from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree


EPS = 1e-5


parser = argparse.ArgumentParser()

# Model
parser.add_argument('--mode', type=str, choices=['mpr', 'diffpool', 'mincut',
                                                 'flat', 'gin', 'avgmlp'])
parser.add_argument('--pagerank_pooling', type=bool, default=True)
parser.add_argument('--cluster_dims', type=int, nargs='+')
parser.add_argument('--hidden_dims', type=int, nargs='+')
parser.add_argument('--interval_overlap', type=float, default=0.1)
parser.add_argument('--pooling_ratio', type=float, default=0.25)
parser.add_argument('--std_hidden_dim', type=int, default=32)

# Optimization
parser.add_argument('--dataset', type=str, default='DD',
                    choices=['COLLAB', 'DD', 'PROTEINS', 'REDDIT-BINARY',
                             'MUTAG', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI',
                             'REDDIT-MULTI-5K'])
parser.add_argument('--fold', type=int, choices=range(11), default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lrate', type=float, default=1e-3)
parser.add_argument('--lrate_anneal_coef', type=float, default=0)
parser.add_argument('--sim_batch_size', type=int, default=32)

# Logs etc
parser.add_argument('--log_dir', type=str, default='./logs/')

args = parser.parse_args()
log_name = '%s_%s_%d_%s_%d_%.4f_%.2f' % (str(args.cluster_dims),
                                         str(args.hidden_dims),
                                         args.interval_overlap,
                                         args.dataset,
                                         args.epochs,
                                         args.lrate,
                                         args.lrate_anneal_coef)


def get_graph_classification_dataset(dataset):
    node_transform = None
    if dataset in ['COLLAB', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI',
                   'REDDIT-MULTI-5K']:
        node_transform = OneHotDegree(max_degree=64)

    path = osp.join(osp.dirname('/tmp/'), dataset)
    dataset = TUDataset(path, name=dataset, pre_transform=node_transform)

    return dataset


def evaluate_loss():
    loss = F.cross_entropy(y_pred, data.y)
    return loss


def update_lrate(optimizer, epoch):
    if args.lrate_anneal_coef and epoch >= args.epochs // 2:
        optimizer.param_groups[0]['lr'] = (optimizer.param_groups[0]['lr'] *
                                           args.lrate_anneal_coef)


def select_subset(this_dataset, indices):
    subset = this_dataset[torch.LongTensor(indices)]
    y = np.concatenate([d.y for d in subset])
    return subset, y


# Load dataset
dataset = get_graph_classification_dataset(args.dataset)
print(args.dataset, dataset[0],
      dataset.num_classes, 'classes',
      dataset.num_features, 'features')

# Determine runtime device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare log file
log_name += '_' + str(datetime.datetime.now()) + '.txt'
if not osp.exists(args.log_dir):
    os.makedirs(args.log_dir)
f = open(osp.join(args.log_dir, log_name), 'w')

# Set seeds (need to maintain same splits and shuffling across models/runs)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=False)

curr_fold = 0
test_accs = []

for train_val_idxs, test_idxs in kf.split(np.zeros(len(dataset.data.y)), dataset.data.y):
    print("Train val idx", len(train_val_idxs))
    print("Test idx", len(test_idxs))

    curr_fold += 1
    if args.fold and curr_fold != args.fold:
        continue
    s = '>>> 10-fold cross-validation --- fold %d' % curr_fold
    print(s)
    f.write(s + '\n')

    # Split into train-val and test
    train_val_dataset, train_val_y = select_subset(dataset, train_val_idxs)
    test_dataset, test_y = select_subset(dataset, test_idxs)

    # Split first set into train and val
    kf2 = StratifiedKFold(n_splits=9, shuffle=False)
    for train_idxs, val_idxs in kf2.split(np.zeros(len(train_val_y)), train_val_y):
        train_dataset, train_y = select_subset(train_val_dataset, train_idxs)
        val_dataset, val_y = select_subset(train_val_dataset, val_idxs)
        break

    # Shuffle the training data
    shuffled_idx = torch.randperm(len(train_dataset))
    train_dataset, train_y = select_subset(train_dataset, shuffled_idx)

    if args.mode == 'mpr':
        model = MPRModel(dataset, args.hidden_dims, args.cluster_dims, args.interval_overlap)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    elif args.mode == 'flat':
        model = FlatModel().to(device)
        print(model)
        f.write(str(model) + '\n')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    elif args.mode == 'gin':
        model = GINModel(hidden_dim=args.std_hidden_dim).to(device)
        print(model)
        f.write(str(model) + '\n')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    elif args.mode == 'avgmlp':
        model = AverageMLP().to(device)
        print(model)
        f.write(str(model) + '\n')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    else:
        model = StandardPoolingModel(mode=args.mode,
                                     hidden_dim=args.std_hidden_dim).to(device)
        print(model)
        f.write(str(model) + '\n')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)

    max_val_acc = 0.0
    for epoch in range(args.epochs):
        # Train model
        train_loss = 0
        if args.mode in ['mpr', 'flat', 'gin', 'avgmlp']:
            model.train()
        else:
            train_loss1 = 0
            train_loss2 = 0
            model.train()

        optimizer.zero_grad()

        for i, data in enumerate(train_dataset):
            data = data.to(device)
            if args.mode == 'mpr':
                y_pred = model(data.x, data.edge_index, args.pagerank_pooling)
                y_pred = y_pred.unsqueeze(0)
            elif args.mode in ['flat', 'gin', 'avgmlp']:
                y_pred = model(data.x, data.edge_index)
            else:
                y_pred, loss1, loss2 = model(data.x, data.edge_index)

            loss = evaluate_loss()
            train_loss += loss
            if args.mode == 'mpr':
                (loss / args.sim_batch_size).backward()
                if i % args.sim_batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            elif args.mode in ['flat', 'gin', 'avgmlp']:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                train_loss1 += loss1
                train_loss2 += loss2
                total_loss = loss + loss1 + loss2
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= len(train_dataset)
        if args.mode in ['diffpool', 'mincut']:
            train_loss1 /= len(train_dataset)
            train_loss2 /= len(train_dataset)

        # Run validation set
        val_acc = 0
        val_loss = 0
        if args.mode in ['mpr', 'flat', 'gin', 'avgmlp']:
            model.eval()
            val_loss = 0
        else:
            model.eval()
            val_loss1 = 0
            val_loss2 = 0

        with torch.no_grad():
            for _, data in enumerate(val_dataset):
                data = data.to(device)
                if args.mode == 'mpr':
                    y_pred = model(data.x, data.edge_index, args.pagerank_pooling)
                    y_pred = y_pred.unsqueeze(0)
                elif args.mode in ['flat', 'gin', 'avgmlp']:
                    y_pred = model(data.x, data.edge_index)
                else:
                    y_pred, loss1, loss2 = model(data.x, data.edge_index)

                loss = evaluate_loss().detach().cpu().numpy()
                val_loss += loss
                if args.mode in ['diffpool', 'mincut']:
                    val_loss1 += loss1
                    val_loss2 += loss2
                val_acc += y_pred.max(1)[1].eq(data.y)

            val_acc = float(val_acc) / len(val_dataset)
            val_loss /= len(val_dataset)
            if args.mode in ['mpr', 'flat', 'gin', 'avgmlp']:
                s = ('Epoch %d - train loss %.4f, val loss %.4f, val accuracy %.4f' %
                     (epoch, train_loss, val_loss, val_acc))
            else:
                val_loss1 /= len(dataset)
                val_loss2 /= len(dataset)
                s = (('Epoch %d - train loss %.4f, loss1 %.4f, loss2 %.4f, '
                      'val loss %.4f, loss1 %.4f, loss2 %.4f, val accuracy %.4f') %
                     (epoch,
                      train_loss, train_loss1, train_loss2,
                      val_loss, val_loss1, val_loss2,
                      val_acc))
            print(s)
            f.write(s + '\n')

            if val_acc > max_val_acc:
                s = 'New best validation accuracy at epoch %d: %.4f' % (epoch, val_acc)
                max_val_acc = val_acc
                print(s)
                f.write(s + '\n')

                # Run test set
                test_acc = 0
                test_loss = 0
                if args.mode in ['mpr', 'flat', 'gin', 'avgmlp']:
                    model.eval()
                else:
                    model.eval()
                    test_loss1 = 0
                    test_loss2 = 0

                for _, data in enumerate(test_dataset):
                    data = data.to(device)
                    if args.mode == 'mpr':
                        y_pred = model(data.x, data.edge_index, args.pagerank_pooling)
                        y_pred = y_pred.unsqueeze(0)
                    elif args.mode in ['flat', 'gin', 'avgmlp']:
                        y_pred = model(data.x, data.edge_index)
                    else:
                        y_pred, loss1, loss2 = model(data.x, data.edge_index)

                    loss = evaluate_loss().detach().cpu().numpy()
                    test_loss += loss
                    if args.mode in ['diffpool', 'mincut']:
                        test_loss1 += loss1
                        test_loss2 += loss2
                    test_acc += y_pred.max(1)[1].eq(data.y)

                test_acc = float(test_acc) / len(test_dataset)
                test_loss /= len(test_dataset)
                if args.mode in ['mpr', 'flat', 'gin', 'avgmlp']:
                    s = ('Epoch %d - test loss %.4f, test accuracy %.4f' %
                         (epoch, test_loss, test_acc))
                else:
                    test_loss1 /= len(dataset)
                    test_loss2 /= len(dataset)
                    s = ((('Epoch %d - test loss %.4f, loss1 %.4f, loss2 %.4f,'
                           'accuracy %.4f') %
                          (epoch, test_loss, test_loss1, test_loss2, test_acc)))
                print(s)
                f.write(s + '\n')

        update_lrate(optimizer, epoch)
    test_accs.append(test_acc)

s = 'Test accuracies: %s, %.4f +- %.4f' % (str(test_accs),
                                           np.mean(np.array(test_accs)),
                                           np.std(np.array(test_accs)))
print(s)
f.write(s + '\n')

f.close()
