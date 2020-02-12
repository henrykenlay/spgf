import logging
import pickle
import warnings

import click
import networkx as nx
import pandas as pd
import pygsp
from tqdm import tqdm

from utilities import khop_subgraph, laplacian_distance, non_pendant_edges, has_isolated_nodes, sample_edges

warnings.filterwarnings("ignore", category=FutureWarning)
logging.disable(logging.CRITICAL)


@click.command()
@click.option("--repeats", type=int, help="Number of times to repeat the experiment")
@click.option("--dataset", type=str, help="What type of graph to perturb (BA or sensor)")
@click.option("--proportion_remove", type=float, help="What proportion of edges to remove from the data")
@click.option("--n", type=int, help="Size of graphs")
@click.option("--m", default=3, type=int, help="Number of edges to attach from a new node to existing nodes in BA graph")
@click.option("--k", default=None, type=int, help="Size of k-hop neighbourbood")
def main(dataset, repeats, proportion_remove, k, n, m):
    graph_generator = GraphGenerator(dataset, n, m)
    if k is None:
        experiment_name = f"{graph_generator}_{proportion_remove}"
    else:
        experiment_name = f"{graph_generator}_{proportion_remove}_{k}"

    print(f"Running experiment {experiment_name}")

    # Generate perturbed graphs
    graphs = []
    dfs = []
    for repeat in tqdm(range(repeats)):
        G = graph_generator.generate()
        remove = int(proportion_remove * G.number_of_edges())
        result = khop_edge_deletion(G, k, remove)
        while result is None:
            G = graph_generator.generate()
            remove = int(proportion_remove * G.number_of_edges())
            result = khop_edge_deletion(G, k, remove)
        Gp, edges, node = result
        graphs.append([G, Gp])
        df = edges_to_df(edges)
        df["khopnode"] = node
        df["repeat"] = repeat
        df["distance"] = laplacian_distance(G, Gp)
        dfs.append(df)
    dfs = pd.concat(dfs, ignore_index=True)

    # Save results
    pickle.dump(graphs, open(f"results/{experiment_name}.p", "wb"))
    dfs.to_csv(f"results/{experiment_name}.csv")


class GraphGenerator:
    def __init__(self, dataset, n, m=None):
        self.dataset = dataset
        self.n = n
        self.m = m

    def generate(self):
        if self.dataset == "BA":
            return nx.barabasi_albert_graph(self.n, self.m)
        elif self.dataset == "Sensor":
            G = pygsp.graphs.Sensor(self.n)
            while not G.is_connected():
                G = pygsp.graphs.Sensor(self.n)
            return nx.Graph(G.W)

    def __repr__(self):
        if self.dataset == "BA":
            return f"BA({self.n},{self.m})"
        elif self.dataset == "Sensor":
            return f"Sensor({self.n})"


def edges_to_df(edges):
    data = []
    for edge in edges:
        data.append([min(edge), max(edge)])
    return pd.DataFrame(data, columns=["u", "v"])


def khop_edge_deletion(G, k, r, max_iter=1000):
    """
    Removes r edges which are in a k-hop neighbourhood of some node, the perturbed graph will not have isolated nodes
    
    If k is None then the samples are taken uniformly
    """
    solution, iteration = None, 0
    while solution is None:
        iteration = iteration + 1
        if iteration == max_iter:
            return None
        subgraph, node = khop_subgraph(G, k) if k is not None else (G, None)
        if len(non_pendant_edges(subgraph)) < r:
            continue
        edges = sample_edges(subgraph, r, non_pendant=True)
        Gp = G.copy()
        Gp.remove_edges_from(edges)
        if not has_isolated_nodes(Gp):
            solution = Gp
    return solution, edges, node


if __name__ == "__main__":
    main()
