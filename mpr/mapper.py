import numpy as np
import torch
import networkx as nx


def generate_intervals(num=5, overlap=0.10, vmin=0, vmax=1):
    """Generates the Mapper intervals in [0, 1] with the specified overlap.

    params:
      num: Number of intevals
      overlap: Percentage specifying overlap between consecutive segments.
    """
    assert 0.0 <= overlap <= 1.0  # Overlap must be a percentage

    overlap *= 1. / (num + 1)
    mid = np.linspace(vmin, vmax, num+1)
    high = mid + overlap
    low = mid - overlap
    intervals = np.concatenate((low[:-1, None], high[1:, None]), axis=1)
    return intervals


def compute_preimages(G, f, intervals=generate_intervals()):
    """Computes the preimages of the function f using the specified intervals."""
    preimages = []
    for interv in intervals:
        preimages.append(np.arange(G.number_of_nodes())[np.logical_and(
            interv[0] <= f, f <= interv[1])])
    return preimages


def mapper_pool(G, preimages, adj):
    """Given a graph G and the preimages of the function f on G it builds the
      mapper graph.

    params:
      G: The graph to be visualized.
      preimages: The preimages of a function f on the nodes of graph G.
    """
    mnode_to_nodes = torch.zeros((len(preimages), G.number_of_nodes())).to(adj.device)

    # Each preimage has a corresponding colour given by its index.
    for i, pre_img in enumerate(preimages):
        # Skip if the preimage is empty
        if not len(pre_img):
            continue

        # Make each connected component a node and assign its color.
        mnode_to_nodes[i, np.fromiter(pre_img, int, len(pre_img))] = 1.0

    # Initialise Mapper graph
    mnode_to_nodes = torch.softmax(mnode_to_nodes, dim=0)
    adj = mnode_to_nodes @ adj @ mnode_to_nodes.t()

    return mnode_to_nodes, adj


def mpr_pool(x, adj, clusters, overlap, EPS=1e-9):
    # Compute the PageRank of the nodes
    graph = nx.from_numpy_matrix(adj.detach().cpu().numpy()).to_undirected()
    pr_dict = nx.pagerank_scipy(graph)
    pr = np.empty(graph.number_of_nodes())

    # Convert dictionary to numpy array
    keys = np.array(list(pr_dict.keys()), dtype=np.int32)
    values = np.array(list(pr_dict.values()), dtype=np.float32)
    pr[keys] = values

    # Scale the lens values in the unit interval
    pr -= np.min(pr)
    pr /= max(np.max(pr), EPS)
    pr = np.ravel(pr)

    # Generate intervals
    curr_cluster_dim = clusters
    intervals = generate_intervals(num=curr_cluster_dim, overlap=overlap)

    # Compute the cluster assignment and the new features
    preimages = compute_preimages(graph, pr, intervals)
    St, A = mapper_pool(graph, preimages, adj)
    X = St @ x

    return X, A