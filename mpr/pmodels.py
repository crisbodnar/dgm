import numpy as np
import torch
import torch.nn.functional as F

from mpr.mapper import mpr_pool
from mpr.utils import dense_diff_pool, dense_mincut_pool, to_dense_adj


from torch_geometric.nn import GCNConv, GINConv
from torch.nn import Linear, ModuleList, BatchNorm1d, Sequential
from torch_geometric.utils import to_networkx, dense_to_sparse


class GEmbedNet(torch.nn.Module):
    def __init__(self, input_dim=89, hidden_dim=128):
        super(GEmbedNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


class GClassifier(torch.nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(GClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mlp = Linear(hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        x = F.dropout(x, training = self.training)
        x = torch.mean(x, dim=0)
        x = self.mlp(x)
        return x


class MPRModel(torch.nn.Module):
    def __init__(self, hidden_dims, num_node_features, cluster_dims):
        super(MPRModel, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_node_features = num_node_features
        self.cluster_dims = cluster_dims
        self.embed_nets = ModuleList()

        prev_dim = num_node_features
        for i in range(len(hidden_dims)):
            curr_dim = hidden_dims[i]
            self.embed_nets.append(GEmbedNet(input_dim=prev_dim, hidden_dim=curr_dim))
            prev_dim = curr_dim

        self.g_classifier = GClassifier(input_dim=hidden_dims[-1], hidden_dim=hidden_dims[-1])

    def forward(self, x, edge_index, edge_attr):
        for i in range(len(self.hidden_dims)):
            x = self.embed_nets[i](x, edge_index)
            x, adj = mpr_pool(x, to_dense_adj(edge_index, x.size(0), edge_attr=edge_attr),
                              clusters=self.cluster_dims[i], overlap=self.overlap)

            edge_index, edge_attr = dense_to_sparse(adj.squeeze(0))

        y_pred = self.g_classifier(x, edge_index, edge_attr)
        return y_pred


class StandardPoolingModel(torch.nn.Module):
    def __init__(self, dataset, pooling_ratio=0.5, mode='diffpool', hidden_dim=32):
        assert mode in ['diffpool', 'mincut']
        super(StandardPoolingModel, self).__init__()
        self.graph_convs = ModuleList([
            GCNConv(dataset.num_node_features, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)])
        self.graph_skips = ModuleList([
            Linear(dataset.num_node_features, hidden_dim),
            Linear(hidden_dim, hidden_dim),
            Linear(hidden_dim, hidden_dim)])

        max_num_nodes = np.max(dataset.data.num_nodes)
        pool_dims = [int(max_num_nodes * pooling_ratio),
                     int(max_num_nodes * pooling_ratio**2)]
        self.mode = mode
        self.pooling_fn = (dense_diff_pool if mode == 'diffpool'
                           else dense_mincut_pool)
        if mode == 'mincut':
            self.assignment_ws = ModuleList([
                Linear(dataset.num_node_features, hidden_dim),
                Linear(hidden_dim, pool_dims[0]),
                Linear(hidden_dim, hidden_dim),
                Linear(hidden_dim, pool_dims[1])])
        else:
            self.pool_convs = ModuleList([
                GCNConv(dataset.num_node_features, pool_dims[0]),
                GCNConv(hidden_dim, pool_dims[1])])

        self.classifier = Linear(hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index):
        total_loss1 = 0
        total_loss2 = 0
        edge_attr = None

        for i in range(len(self.graph_convs)):
            if i < len(self.graph_convs) - 1:
                if self.mode == 'mincut':
                    s = self.assignment_ws[2*i+1](F.relu(
                        self.assignment_ws[2*i](x)))
                else:
                    s = self.pool_convs[i](x, edge_index, edge_attr)

            x = F.relu((self.graph_convs[i](x, edge_index, edge_attr) +
                        self.graph_skips[i](x)))

            if i < len(self.graph_convs) - 1:
                x, adj, loss1, loss2 = self.pooling_fn(
                    x,
                    to_dense_adj(edge_index, x.size(0), edge_attr=edge_attr),
                    s)
                edge_index, edge_attr = dense_to_sparse(adj.squeeze(0))
                x = x.squeeze(0)
                total_loss1 += loss1
                total_loss2 += loss2

        x_avg = torch.mean(x, dim=0).unsqueeze(0)
        out = self.classifier(x_avg)

        return out, total_loss1, total_loss2


class FlatModel(torch.nn.Module):
    def __init__(self, hidden_dim=32):
        super(FlatModel, self).__init__()
        self.graph_convs = ModuleList([
            GCNConv(dataset.num_node_features, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)])
        self.graph_skips = ModuleList([
            Linear(dataset.num_node_features, hidden_dim),
            Linear(hidden_dim, hidden_dim)])

        self.classifier = Linear(hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index):
        for i in range(len(self.graph_convs)):
            x = F.relu((self.graph_convs[i](x, edge_index) +
                        self.graph_skips[i](x)))

        x_avg = torch.mean(x, dim=0).unsqueeze(0)
        out = self.classifier(x_avg)

        return out


class GINModel(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super(GINModel, self).__init__()
        self.gin_layers = ModuleList([])
        input_dim = dataset.num_node_features
        for i in range(5):
            mlp = Sequential(
                Linear(input_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim)
            )
            self.gin_layers.append(GINConv(nn=mlp))
            input_dim = hidden_dim

        self.classifier = Linear(5 * hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index):
        reprs = []
        for i in range(5):
            x = self.gin_layers[i](x, edge_index)
            reprs.append(torch.sum(x, dim=0))

        summary = torch.cat(reprs, dim=0).unsqueeze(0)
        out = self.classifier(summary)

        return out


class AverageMLP(torch.nn.Module):
    def __init__(self):
        super(AverageMLP, self).__init__()
        self.classifier = Linear(dataset.num_node_features, dataset.num_classes)

    def forward(self, x, edge_index):
        summary = torch.mean(x, dim=0).unsqueeze(0)
        out = self.classifier(summary)

        return out