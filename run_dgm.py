import numpy as np

from dgm.dgm import *
from dgm.plotting import *
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import to_networkx


def plot_sdgm_spam_binary():
    print("Plotting SDGM Spam graph")

    # Load the GCN predicted probabilties on the spammer dataset.
    # These are loaded directly from a file here to make running the code easier.
    spam_prob = np.load('./data/spam_prob.npy')
    spam_prob = spam_prob[:, 1][:, None]

    # Load the spam graph in networkx.
    spam_graph = nx.read_gpickle('./data/spam_graph.gpickle')

    print("G nodes", spam_graph.number_of_nodes())
    print("G edges", spam_graph.number_of_edges())

    spam_sdgm, res = build_dgm_graph(spam_graph, spam_prob, num_intervals=20, overlap=0.0, eps=0.01,
                                     min_component_size=4, sdgm=True)

    print("fMG nodes", spam_sdgm.number_of_nodes())
    print("fMG edges", spam_sdgm.number_of_edges())

    # Plot the SDGM graph
    node_size = np.array([len(cc) for _, cc in res['mnode_to_nodes'].items()])
    plot_graph(spam_sdgm, node_color=res['mnode_to_color'], node_size=node_size, edge_weight=res['edge_weight'],
               node_list=res['node_list'], name='sdgm_spam_prob')

    # Plot the SDGM graph with true labels
    labels = np.load('./data/spam_label.npy')
    labeled_colors = color_mnodes_with_labels(res['mnode_to_nodes'], labels, binary=True)

    plot_graph(spam_sdgm, node_color=labeled_colors, node_size=node_size, edge_weight=res['edge_weight'],
               node_list=res['node_list'], name='sdgm_spam_label', colorbar=True)


def plot_sdgm_cora_dgi():
    print("Plotting SDGM Cora graph")

    # Load the cora dataset
    data = Planetoid(root='/tmp/Cora', name='Cora')[0]

    # Load the DGI + t-SNE embeddings
    embed = np.load('./data/cora_dgi_tsne.npy')

    # Load the Cora graph
    cora_graph = to_networkx(data).to_undirected()

    print("G nodes", cora_graph.number_of_nodes())
    print("G edges", cora_graph.number_of_edges())

    cora_sdgm, res = build_dgm_graph(cora_graph, embed, num_intervals=40, overlap=0.0, eps=0.07,
                                     min_component_size=20, sdgm=True)

    node_color = color_from_bivariate_data(res['mnode_to_color'][:, 0], res['mnode_to_color'][:, 1])
    node_size = np.array([len(cc) for _, cc in res['mnode_to_nodes'].items()])
    plot_graph(cora_sdgm, node_color=node_color, node_size=node_size, edge_weight=res['edge_weight'],
               node_list=res['node_list'], name='sdgm_cora_dgi_embed', colorbar=False)

    print("fMG nodes", cora_sdgm.number_of_nodes())
    print("fMG edges", cora_sdgm.number_of_edges())

    labeled_colors = color_mnodes_with_labels(res['mnode_to_nodes'], data.y.cpu().numpy(), binary=False)
    plt.set_cmap(cm.Accent)
    plot_graph(cora_sdgm, node_color=labeled_colors, node_size=node_size, edge_weight=res['edge_weight'],
               node_list=res['node_list'], name='sdgm_cora_dgi_embed_labeled', colorbar=False)


def plot_dgm_spam_binary():
    print("Plotting DGM Spam graph")

    # Load the GCN predicted probabilties on the spammer dataset.
    # These are loaded directly from a file here to make running the code easier.
    spam_prob = np.load('./data/spam_prob.npy')
    spam_prob = spam_prob[:, 1][:, None]

    # Load the spam graph in networkx.
    spam_graph = nx.read_gpickle('./data/spam_graph.gpickle')

    print("G nodes", spam_graph.number_of_nodes())
    print("G edges", spam_graph.number_of_edges())

    spam_sdgm, res = build_dgm_graph(spam_graph, spam_prob, num_intervals=10, overlap=0.2, eps=0.00,
                                     min_component_size=60, sdgm=False)

    # Plot the SDGM graph
    node_size = np.array([len(cc) for _, cc in res['mnode_to_nodes'].items()])
    plot_graph(spam_sdgm, node_color=res['mnode_to_color'], node_size=node_size, edge_weight=res['edge_weight'],
               node_list=res['node_list'], name='dgm_spam_prob')

    # Plot the SDGM graph with true labels
    labels = np.load('./data/spam_label.npy')
    labeled_colors = color_mnodes_with_labels(res['mnode_to_nodes'], labels, binary=True)

    print("fMG nodes", spam_sdgm.number_of_nodes())
    print("fMG edges", spam_sdgm.number_of_edges())

    plot_graph(spam_sdgm, node_color=labeled_colors, node_size=node_size, edge_weight=res['edge_weight'],
               node_list=res['node_list'], name='dgm_spam_label', colorbar=True)


def plot_dgm_cora_dgi():
    print("Plotting DGM Cora graph")

    # Load the cora dataset
    data = Planetoid(root='/tmp/Cora', name='Cora')[0]

    # Load the DGI + t-SNE embeddings
    embed = np.load('./data/cora_dgi_tsne.npy')

    # Load the Cora graph
    cora_graph = to_networkx(data).to_undirected()

    print("G nodes", cora_graph.number_of_nodes())
    print("G edges", cora_graph.number_of_edges())

    cora_sdgm, res = build_dgm_graph(cora_graph, embed, num_intervals=10, overlap=0.2, eps=0.00,
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


def main():
    # Plots the spammer graph with Structural Deep Graph Mapper using a binary GCN classifier as lens.
    plot_sdgm_spam_binary()

    # Plots Structural Deep Graph Mapper with (unsupervised) DGI lens for the Cora graph.
    plot_sdgm_cora_dgi()

    # Plots the spammer graph with Deep Graph Mapper using a binary GCN classifier as lens.
    plot_dgm_spam_binary()

    # Plots Deep Graph Mapper with (unsupervised) DGI lens for the Cora graph.
    plot_dgm_cora_dgi()


if __name__ == "__main__":
    main()