"""Data reading utils."""

import json
import glob
import torch
import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import re

from tqdm.notebook import tqdm
from texttable import Texttable
from scipy import integrate


def merge_metalabel(meta_data):
    for i in range(len(meta_data)):
        if meta_data[i]["label"][0] < 2:
            meta_data[i]["label"][0] = 1
        elif meta_data[i]["label"][0] > 2:
            meta_data[i]["label"][0] = 3
        elif meta_data[i]["label"][0] == 2:
            pass
        else:
            print("WRONG meta labels!!")
    return meta_data


def load_mergeddata(data):
    G = nx.from_edgelist(data["edges"])
    features = data["node_feature"]
    features = np.array(features, dtype=np.float32)
    features = torch.tensor(features)
    target = data["target"]
    # merge target labels
    # NC={CN,SMC}={0,1}=1, EMCI=2, AD={LMCI,AD}={3,4}=3
    if target < 2:
        target = 1
    elif target > 2:
        target = 3
    elif target == 2:
        target = 2
    else:
        print(f"WRONG label!!")
    target = torch.tensor([target])
    return G, features, target


def load_data(data):
    G = nx.from_edgelist(data["edges"])
    features = data["node_feature"]
    features = np.array(features, dtype=np.float32)
    features = torch.tensor(features)
    target = data["target"]
    if target <= 1:  # 0 NC, 1 SMC, 2 EMCI, 3 LMCI, 4 AD
        target = 1  # set label 0,1 = 1
    elif (
        target > 1
    ):  # set label 2, 3, 4 = 0, file dataset_meta_NC-AD do not have label 2
        target = 0
    target = torch.tensor([target])
    return G, features, target


def summary(arr, axis=None):
    # print("shape {}, max {}, min {}, mean {}, std {}".format(arr.shape, np.max(arr),np.min(arr),np.mean(arr),np.std(arr)))
    return {
        "shape": arr.shape,
        "max": np.max(arr, axis=axis),
        "min": np.min(arr, axis=axis),
        "mean": np.mean(arr, axis=axis),
        "std": np.std(arr, axis=axis),
        "sum": np.sum(arr, axis=axis),
    }


def display_hist(data, bins=10, figsize=(10, 6), title=None):
    ax = data.plot.hist(figsize=figsize, bins=bins, edgecolor='white', linewidth=1.2)
    #     ax.set_title(title)
    for p in ax.patches:
        if p.get_height() != 0:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.02))
    plt.show()


def rowwise_rescale(mx):
    mx_max = mx.max(axis=1)
    mx_max = mx_max[:, np.newaxis]
    assert np.sum(mx.min(axis=1)) == 0, "Causion!! Min not zero"
    mx = mx / mx_max
    return mx


def standardize(mx):
    mx = (mx - mx.mean()) / mx.std()
    mx = mx - mx.min()
    return mx


def sparse_to_nxtuple(mx):
    if not scipy.sparse.isspmatrix_coo(mx):
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col, mx.data)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape


def disparity_filter(G, weight='weight'):
    '''Compute significance scores (alpha) for weighted edges in G as defined in Serrano et al. 2009
    Args
        G: Weighted NetworkX graph
    Returns
        Weighted graph with a significance score (alpha) assigned to each edge
    References
        M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks. PNAS, 106:16, pp. 6483-6488.
    '''
    # undirected case
    B = nx.Graph()
    if nx.number_of_selfloops(G) != 0:
        G.remove_edges_from(G.selfloop_edges())  # remove self loops
    for u in G:  # iterate through nodes
        k = len(G[u])  # how many nodes connected with u
        if k >= 1:
            sum_w = sum(np.absolute(G[u][v][weight]) for v in G[u])
            for v in G[u]:
                w = G[u][v][weight]
                p_ij = float(np.absolute(w)) / sum_w
                alpha_ij = (
                    1
                    - (k - 1) * integrate.quad(lambda x: (1 - x) ** (k - 2), 0, p_ij)[0]
                )
                B.add_edge(u, v, weight=w, alpha=float('%.4f' % alpha_ij))
        else:
            B.add_node(u)
    return B


def disparity_filter_alpha_cut(G, weight='weight', alpha_t=0.4, cut_mode='or'):
    '''Performs a cut of the graph previously filtered through the disparity_filter function.

    Args
    ----
    G: Weighted NetworkX graph

    weight: string (default='weight')
        Key for edge data used as the edge weight w_ij.

    alpha_t: double (default='0.4')
        The threshold for the alpha parameter that is used to select the surviving edges.
        It has to be a number between 0 and 1.

    cut_mode: string (default='or')
        Possible strings: 'or', 'and'.
        It works only for directed graphs. It represents the logic operation to filter out edges
        that do not pass the threshold value, combining the alpha_in and alpha_out attributes
        resulting from the disparity_filter function.


    Returns
    -------
    B: Weighted NetworkX graph
        The resulting graph contains only edges that survived from the filtering with the alpha_t threshold

    References
    ---------
    .. M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks. PNAS, 106:16, pp. 6483-6488.
    '''

    if nx.is_directed(G):  # Directed case:
        B = nx.DiGraph()
        for u, v, w in G.edges(data=True):
            try:
                alpha_in = w['alpha_in']
            except KeyError:  # there is no alpha_in, so we assign 1. It will never pass the cut
                alpha_in = 1
            try:
                alpha_out = w['alpha_out']
            except KeyError:  # there is no alpha_out, so we assign 1. It will never pass the cut
                alpha_out = 1

            if cut_mode == 'or':
                if alpha_in < alpha_t or alpha_out < alpha_t:
                    B.add_edge(u, v, weight=w[weight])
            elif cut_mode == 'and':
                if alpha_in < alpha_t and alpha_out < alpha_t:
                    B.add_edge(u, v, weight=w[weight])
        return B

    else:
        B = nx.Graph()  # Undirected case:
        if nx.number_of_selfloops(G) != 0:
            G.remove_edges_from(G.selfloop_edges())  # remove self loops
        for u in G:  # iterate through nodes
            k = len(G[u])  # how many nodes connected with u
            if k >= 1:
                count_added_edges = 0
                for v in G[u]:
                    w = G[u][v]
                    try:
                        alpha = w['alpha']
                    except KeyError:  # there is no alpha, so we assign 1. It will never pass the cut
                        alpha = 1
                    if alpha < alpha_t:
                        B.add_edge(u, v, weight=w[weight])
                        count_added_edges += 1
                if count_added_edges == 0:  # no edges added to node u
                    B.add_node(u)
            else:
                B.add_node(u)  # if no node connected to u
        return B


def read_node_labels(args):
    """
    Reading the graphs from disk.
    :param args: Arguments object.
    :return identifiers: Hash table of unique node labels in the dataset.
    :return class_number: Number of unique graph classes in the dataset.
    """
    print("\nCollecting unique node labels.\n")
    labels = set()
    targets = set()
    graphs = glob.glob(args.data_folder + "*.json")
    #     try:
    #         graphs = graphs + glob.glob(args.test_graph_folder + "*.json")
    #     except:
    #         pass
    for g in tqdm(graphs):
        data = json.load(open(g))
        labels = labels.union(set(list(data["labels"].values())))
        targets = targets.union(set([data["target"]]))
    identifiers = {label: i for i, label in enumerate(list(labels))}
    class_number = len(targets)
    print("\n\nThe number of graph classes is: " + str(class_number) + ".\n")
    return identifiers, class_number


def create_features(data, identifiers):
    """
     Creates a tensor of node features.
    :param data: Hash table with data.
    :param identifiers: Node labels mapping.
    :return graph: NetworkX object.
    :return features: Feature Tensor (PyTorch).
    """
    graph = nx.from_edgelist(data["edges"])
    features = []
    for node in graph.nodes():
        features.append(
            [
                1.0 if data["labels"][str(node)] == i else 0.0
                for i in range(len(identifiers))
            ]
        )
    features = np.array(features, dtype=np.float32)
    features = torch.tensor(features)
    return graph, features


def create_batches(graphs, batch_size):
    """
    Creating batches of graph locations.
    :param graphs: List of training graphs.
    :param batch_size: Size of batches.
    :return batches: List of lists with paths to graphs.
    """
    batches = [graphs[i : i + batch_size] for i in range(0, len(graphs), batch_size)]
    return batches


def calculate_reward(target, prediction):
    """
    Calculating a reward for a prediction.
    :param target: True graph label.
    :param prediction: Predicted graph label.
    """
    reward = target == torch.argmax(prediction)
    #     reward = 2*(reward.float()-0.5)
    reward = reward.float() - 1.0
    return reward


def calculate_gnnexplainer_reward(target, prediction):
    """
    Calculating a gnnexplainer reward for a prediction.
    :param target: True graph label.
    :param prediction: Predicted graph label.
    :param mask: node feature
    """
    reward = target == torch.argmax(prediction)
    #     reward = 2*(reward.float()-0.5)
    reward = reward.float() - 1.0
    return reward


def calculate_predictive_loss(target, predictions):
    """
    Prediction loss calculation.
    :param data: Hash with label.
    :param prediction: Predicted label.
    :return target: Target tensor.
    :prediction loss: Loss on sample.
    """

    prediction_loss = torch.nn.functional.nll_loss(predictions, target)
    return prediction_loss


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def corrcoef(X):
    avg = X.mean(-1)
    X = X - avg[..., None]
    X_T = X.swapaxes(-2, -1)
    c = X @ X_T
    d = c.diagonal(0, -2, -1)
    stddev = np.sqrt(d)
    c /= stddev[..., None]
    c /= stddev[..., None, :]
    np.clip(c, -1, 1, out=c)
    return c


def corrcoef_diff(X, Y):
    avg = X.mean(-1)
    X = X - avg[..., None]
    avg = Y.mean(-1)
    Y = Y - avg[..., None]
    Y_T = Y.swapaxes(-2, -1)
    c = X @ Y_T
    d = c.diagonal(0, -2, -1)
    stddev = np.sqrt(d)
    c /= stddev[..., None]
    c /= stddev[..., None, :]
    np.clip(c, -1, 1, out=c)
    return c


def cov(X):
    avg = X.mean(-1, keepdims=True)
    X = X - avg
    X_T = X.swapaxes(-2, -1)
    c = X @ X_T / (X.shape[-1] - 1)
    return c


def sliding_window(X, window, step=1, padding=True):
    """
    计算滑动窗口

    Parameters
    ----------
    X: 输入张量
    window: 窗口大小
    padding: 是否填充

    Returns
    -------
    给定数据的所有窗口
    """
    if padding:
        left = (window - 1) // 2
        right = window - 1 - left
        X = np.concatenate((X[..., :left], X, X[..., -right:]), axis=-1)
    X_window = sliding_window_view(X, (X.shape[-2], window), (-2, -1)).squeeze(0)[
        ::step
    ]
    return X_window


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
