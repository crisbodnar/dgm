import argparse

from dgm.dgm import DGM
from dgm.plotting import *
from dgm.utils import *

from torch_geometric.utils.convert import to_networkx

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Dataset to use (spam, cora)', type=str, default='cora')
parser.add_argument('--sdgm', help='Whether to use SDGM or not', action="store_true")
parser.add_argument('--train_mode', help='Supervised or unsupervised training', default='supervised')
parser.add_argument('--reduce_method', help='Method to use for dimensionality reduction', default='tsne')
parser.add_argument('--reduce_dim', help='The final embedding dimension after dimensionality reduction', type=int,
                    default=2)
parser.add_argument('--intervals', help='Number of intervals to use across each axis for the grid', type=int,
                    required=True)
parser.add_argument('--overlap', help='Overlap percentage between consecutive intervals on each axis', type=float,
                    required=True)
parser.add_argument('--eps', help='Edge filtration value for SDGM', type=float, default=0.0)
parser.add_argument('--min_component_size', help='Minimum connected component size to be included in the visualisation',
                    type=int, default=0.0)
parser.add_argument('--dir', help='Directory inside plots where to save the results', default='')
parser.add_argument('--lens', help='Type of graph theoretic lens to use', type=str, required=True)
parser.add_argument('--scale', help='Scale parameter for the density lens', type=float,  default=1.0)
parser.add_argument('--cutoff', type=int)


def get_distance_matrix(graph, cutoff):
    length = dict(nx.all_pairs_shortest_path_length(graph, cutoff=cutoff))

    n = graph.number_of_nodes()
    d = np.ones((n, n), dtype=np.int16) * n

    print("Array created")
    np.fill_diagonal(d, 0.0)
    for n1 in range(n):
        for n2 in length[n1]:
            d[n1, n2] = int(length[n1][n2])
    return d


def plot_mapper_graph(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Plotting the {} graph".format(args.dataset))

    data, num_classes, legend_dict = load_dataset(args.dataset)
    data = data.to(device)
    graph = to_networkx(data).to_undirected()

    print("Graph nodes", graph.number_of_nodes())
    print("Graph edges", graph.number_of_edges())

    if args.lens == 'PR':
        pr = nx.pagerank(graph)
        embed = np.empty(len(pr))
        for node, score in pr.items():
            embed[node] = score
        embed = embed[:, None]
    elif args.lens == 'density':
        d_path = "./data/{}_distance_matrix.npy".format(args.dataset)
        if os.path.isfile(d_path):
            print('Using existing embedding')
            d = np.load(d_path)
        else:
            d = get_distance_matrix(graph, args.cutoff)
            np.save(d_path, d)
        embed = np.sum(np.exp(-d / args.scale), axis=-1)[:, None]
    else:
        raise ValueError('Unsupported lens {}'.format(args.lens))

    embed = reduce_embedding(embed, reduce_dim=args.reduce_dim, method=args.reduce_method)

    print('Creating visualisation...')
    colorbar = embed.shape[1] == 1
    mapper_graph, res = DGM(num_intervals=args.intervals, overlap=args.overlap, eps=args.eps,
                            min_component_size=args.min_component_size, sdgm=args.sdgm).fit_transform(graph, embed)

    plot_graph(mapper_graph, node_color=res['mnode_to_color'], node_size=res['node_sizes'], edge_weight=res['edge_weight'],
               node_list=res['node_list'], name=dgm_name_from_args(args, False), save_dir=args.dir, colorbar=colorbar)

    print("Filtered Mapper Graph nodes", mapper_graph.number_of_nodes())
    print("Filtered Mapper Graph edges", mapper_graph.number_of_edges())
    print("Nodes from the original graph", np.sum(res['node_sizes']))

    binary_labels = (args.reduce_method == 'binary_prob')
    labeled_colors = color_mnodes_with_labels(res['mnode_to_nodes'], data.y.cpu().numpy(), binary=binary_labels)
    plot_graph(mapper_graph, node_color=labeled_colors, node_size=res['node_sizes'], edge_weight=res['edge_weight'],
               node_list=res['node_list'], name=dgm_name_from_args(args, True), save_dir=args.dir, colorbar=binary_labels,
               legend_dict=legend_dict)


if __name__ == "__main__":
    random.seed(444)
    np.random.seed(444)
    torch.manual_seed(444)
    plot_mapper_graph(parser.parse_args())
