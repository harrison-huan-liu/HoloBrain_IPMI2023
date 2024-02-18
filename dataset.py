import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from utils import sorted_aphanumeric, sliding_window, corrcoef
import os
import pickle
from wavelets import cfc, thresholding, harmonic_wavelets
from itertools import starmap
import numpy as np
import time


class GraphSeqDataset(Dataset):
    def __init__(
        self,
        path,
        window,
        step,
        padding,
        ratio,
        wavelets_num,
        beta,
        gamma,
        max_iter,
        min_err,
        node_select,
    ):
        super().__init__()
        self.files = sorted_aphanumeric(
            os.path.join(path, file) for file in os.listdir(path)
        )
        self.window = window
        self.step = step
        self.padding = padding
        self.ratio = ratio
        self.wavelets_num = wavelets_num
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.min_err = min_err
        self.node_select = node_select

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            bold, label = pickle.load(f)
        bold = np.float32(bold)
        bolds = sliding_window(
            bold, window=self.window, step=self.step, padding=self.padding
        )
        fcs = corrcoef(bolds)
        adjs = thresholding(fcs, self.files[idx], ratio=self.ratio)
        graphs = list(map(nx.from_numpy_array, adjs))
        wavelets = [
            harmonic_wavelets(
                adj,
                wavelets_num=self.wavelets_num,
                beta=self.beta,
                gamma=self.gamma,
                max_iter=self.max_iter,
                min_err=self.min_err,
                node_select=self.node_select,
            )
            for adj in adjs
        ]
        cfcs = [
            torch.Tensor(cfc(wavelet, bold)[0])
            for wavelet, bold in zip(wavelets, bolds)
        ]
        # cfcs = np.zeros((10, 10))
        # adjs = torch.rand(4, self.args.node_number, self.args.node_number)
        # adjs[adjs < 0.4] = 0
        # graphs = list(map(nx.from_numpy_array, adjs.numpy()))
        # cfcs = torch.rand(
        #     4, self.args.node_number, self.args.node_dim, self.args.node_dim
        # )
        # cfcs = cfcs @ cfcs.transpose(-2, -1)
        filename_head_tail = os.path.split(self.files[idx])
        return graphs, cfcs, torch.tensor([label] * len(fcs)), filename_head_tail[1]


class GraphDataset(Dataset):
    def __init__(
        self,
        path,
        ratio=0.4,
        wavelets_num=10,
        beta=0.1,
        gamma=0.1,
        max_iter=100,
        min_err=1,
        node_select=90,
    ):
        super().__init__()
        self.files = sorted_aphanumeric(
            os.path.join(path, file) for file in os.listdir(path)
        )
        self.ratio = ratio
        self.wavelets_num = wavelets_num
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.min_err = min_err
        self.node_select = node_select

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            bold, label = pickle.load(f)
        bolds = np.float32(bold)[None, ...]
        fcs = corrcoef(bolds)
        adjs = thresholding(fcs)#,, self.files[idx] ratio=self.ratio
        graphs = list(map(nx.from_numpy_array, adjs))
        # wavelets = [
        #     harmonic_wavelets(
        #         adj,
        #         wavelets_num=self.wavelets_num,
        #         beta=self.beta,
        #         gamma=self.gamma,
        #         max_iter=self.max_iter,
        #         min_err=self.min_err,
        #         node_select=self.node_select,
        #     )
        #     for adj in adjs
        # ]
        # cfcs = [
        #     torch.Tensor(cfc(wavelet, bold)[0])
        #     for wavelet, bold in zip(wavelets, bolds)
        # ]
        cfcs=fcs
        labels = torch.tensor([label] * len(graphs))
        filename_head_tail = os.path.split(self.files[idx])
        return graphs, cfcs, torch.tensor([label] * len(fcs)), filename_head_tail[1], bolds, labels


class GraphSeqFileDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.files = sorted_aphanumeric(
            os.path.join(path, file) for file in os.listdir(os.path.abspath(path))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            d = pickle.load(f)
        graphs = d['graph']
        cfcs = d['cfc']
        labels = torch.tensor([d['label']] * len(graphs))
        filename_head_tail = os.path.split(self.files[idx])

        return graphs, cfcs, labels, filename_head_tail[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GAM.")
    parser.add_argument(
        "--data-folder",
        nargs="?",
        default="./data/task2",
        help="Data graphs folder.",
    )
    parser.add_argument(
        "--node_number",
        type=int,
        default=268,
        help="The number of nodes depends on the input data. Default is 268.",
    )
    parser.add_argument(
        "--node_dim",
        type=int,
        default=10,
        help="Dimensions for node features. Default is 10.",
    )
    args = parser.parse_args()
    graph_seq_dataset = GraphSeqDataset(args.data_folder)
    for x in graph_seq_dataset:
        print(x)
        for graph, cfc_mat, label in zip(*x):
            print(cfc_mat)
            break
        break
