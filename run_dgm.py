import argparse
import torch
import os

from dgm.dgm import *
from dgm.plotting import *
from torch_geometric.utils.convert import to_networkx

from dgm.utils import load_dataset
from dgm.models import GraphClassifier, DGILearner

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Dataset to use (spam, cora)', type=str, default='cora')
parser.add_argument('--method', help='Version to use (SDGM, DGM)', default='SDGM')
parser.add_argument('--train_mode', help='Supervised or unsupervised training', default='supervised')
parser.add_argument('--reduce', help='Method to use for dimensionality reduction', default='tsne')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(dataset, train_mode, device):
    dataset = dataset.to(device)
    if train_mode == 'supervised':
        model = GraphClassifier(dataset.num_node_features, 7, device)
    elif train_mode == 'unsupervised':
        model = DGILearner(dataset.num_node_features, 7, device)
    else:
        raise ValueError('Unsupported train mode {}'.format(train_mode))

    for epoch in range(1, 40):
        train_loss = model.train(dataset)
        test_loss = model.test(dataset)[0]

        if epoch % 5 == 0:
            log = 'Epoch: {:03d}, train_loss: {:.3f}, test_loss:{:.3f}'
            print(log.format(epoch, train_loss, test_loss))

    return model.embed(dataset).detach().cpu().numpy()


def plot_dgm_graph(args):
    print("Plotting the {} graph".format(args.dataset))

    # Load the cora dataset and graph
    data = load_dataset(args.dataset).to(device)
    graph = to_networkx(data).to_undirected()

    print("Graph nodes", graph.number_of_nodes())
    print("Graph edges", graph.number_of_edges())

    # Load the DGI + t-SNE embeddings
    embed_path = "data/{}_{}".format(args.dataset, args.train_mode)
    if os.path.isfile(embed_path):
        print('Using existing embedding')
        embed = np.load(embed_path)
    else:
        print('No embedding found. Training a new model...')
        embed = train_model(data, args.train_mode, device)
        np.save(embed_path, embed)

    embed = reduce_embedding(embed, components=2, method='tsne')

    print('Creating visualisation...')
    cora_sdgm, res = build_dgm_graph(graph, embed, num_intervals=10, overlap=0.2, eps=0.00,
                                     min_component_size=20, sdgm=False)

    node_color = color_from_bivariate_data(res['mnode_to_color'][:, 0], res['mnode_to_color'][:, 1])
    node_size = np.array([len(cc) for _, cc in res['mnode_to_nodes'].items()])
    plot_graph(cora_sdgm, node_color=node_color, node_size=node_size, edge_weight=res['edge_weight'],
               node_list=res['node_list'], name='dgm_cora_dgi_embed', colorbar=False)

    print("fMG nodes", cora_sdgm.number_of_nodes())
    print("fMG edges", cora_sdgm.number_of_edges())

    labeled_colors = color_mnodes_with_labels(res['mnode_to_nodes'], data.y.cpu().numpy(), binary=False)
    plt.set_cmap(cm.Accent)
    plot_graph(cora_sdgm, node_color=labeled_colors, node_size=node_size, edge_weight=res['edge_weight'],
               node_list=res['node_list'], name='dgm_cora_dgi_embed_labeled', colorbar=False)


if __name__ == "__main__":
    plot_dgm_graph(parser.parse_args())
