import networkx as nx
import numpy as np
import scipy.sparse as sp


def sample_edges(G, r, non_pendant=True):
    "Samples r edges from a graph G"
    if non_pendant:
        edges = non_pendant_edges(G)
    else:
        edges = list(G.edges)
    return [edges[i] for i in np.random.choice(range(len(edges)), r, replace=False)]


def non_pendant_edges(G):
    "Returns all non pendant edges of a graph `G`"
    edges = list(G.edges())
    edges = [edge for edge in edges if not is_pendant(G, edge)]
    return edges


def is_pendant(G, edge):
    "Checks if `edge` is pendant in the graph `G`"
    if G.degree(edge[0]) == 1 or G.degree(edge[1]) == 1:
        return True
    else:
        return False


def laplacian_distance(G, Gp):
    r"Calculates $|| \mathcal{L} -  \mathcal{L}_p ||$ using the matrix 2-norm"
    L, Lp = nx.normalized_laplacian_matrix(G), nx.normalized_laplacian_matrix(Gp)
    E = Lp - L
    return sparse_2norm(E)


def sparse_2norm(A):
    "Returns the matrix 2-norm of a sparse matrix `A`"
    return np.abs(sp.linalg.eigsh(A, k=1, which="LM", return_eigenvectors=False))[0]


def has_isolated_nodes(G):
    "Checks if the graph `G` has isolated nodes"
    if len(list(nx.isolates(G))) > 0:
        return True
    else:
        return False


def khop_subgraph(G, k, node=None):
    "Returns hte k-hop subgraph of G around node"
    if node is None:
        node = sample_node(G)
    khop = khop_neighbourhood(G, node, k)
    return G.subgraph(khop), node


def khop_neighbourhood(G, node, k):
    "Returns the k-hop neighbourhood of `node`"
    nodes = set([node])
    for _ in range(k):
        nodes = dilate(G, nodes)
    return nodes


def dilate(G, nodes):
    "Returns the union of the open neighbourhood of all nodes in `nodes`"
    dilation = nodes.copy()
    for node in nodes:
        for neighbour in G.neighbors(node):
            dilation.add(neighbour)
    return dilation


def sample_node(G):
    "Uniformly samples a single node"
    nodes = G.nodes()
    return np.random.choice(list(nodes))
