"""Graph Attention Mechanism."""
import os

from spdsru import SPDLSTM, SPDSRU
from dataset import GraphSeqDataset, GraphSeqFileDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import codecs, json
import torch, torch.nn, torch.nn.functional, torch.optim
import random

import numpy as np
import pandas as pd
import networkx as nx

from datetime import datetime
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

from utils import merge_metalabel, load_mergeddata, load_data
from utils import calculate_reward, calculate_predictive_loss, calculate_gnnexplainer_reward
from utils import (
    read_node_labels,
    create_features,
    create_batches,
    AverageMeter,
)
from spdnet import SPDVectorize, SPDLogMap
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from math import inf
import logging
import joblib
import pickle
from generator import save_data_singledata
from texttable import Texttable

from torch.nn.functional import cross_entropy
from sklearn.metrics import confusion_matrix


"""Parameter parsing."""

import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it learns on the Erdos-Renyi dataset.
    The default hyperparameters give good results without cross-validation.
    """
    parser = argparse.ArgumentParser(description="Run GAM.")

    parser.add_argument(
        '--mode',
        nargs="?",
        default="seq",
        help='experiment mode, ["all", "firstscan", "seq"]',
    )

    parser.add_argument('--debug', action='store_true', default=False, help='Debuging ')

    parser.add_argument(
        '--early_stop',
        nargs="?",
        default=50,
        help='Stop training if loss is not decrease.',
    )

    parser.add_argument(
        '--cuda', action='store_true', default=False, help='CUDA training.'
    )

    parser.add_argument('--seed', type=int, default=111, help='Random seed.')

    parser.add_argument(
        "--data_step",
        type=int,
        default=15,
        help="The step for sliding window to generate graphs.",
    )

    parser.add_argument(
        "--data_window",
        type=int,
        default=30,
        help="The window size for sliding window to generate graphs.",
    )

    parser.add_argument(
        "--wavelets_num",
        type=int,
        default=10,
        help="The wavelets_num for harmonic_wavelets.",
    )

    parser.add_argument(
        "--wavelets_beta",
        type=float,
        default=1,
        help="The beta for harmonic_wavelets.",
    )

    parser.add_argument(
        "--wavelets_gamma",
        type=float,
        default=0.005,
        help="The gamma for harmonic_wavelets.",
    )

    parser.add_argument(
        "--wavelets_max_iter",
        type=int,
        default=1000,
        help="The max_iter for harmonic_wavelets.",
    )

    parser.add_argument(
        "--wavelets_min_err",
        type=float,
        default=0.0001,
        help="The min_err for harmonic_wavelets.",
    )

    parser.add_argument(
        "--wavelets_node_select",
        type=int,
        default=10,
        help="The node_select for harmonic_wavelets.",
    )

    parser.add_argument(
        "--thresholding_ratio",
        type=float,
        default=0.8,
        help="The ratio for thresholding.",
    )

    parser.add_argument(
        "--data_padding",
        action='store_true',
        default=True,
        help="Pad data to get the full window at every time point when sliding window",
    )

    parser.add_argument(
        "--data_folder",
        nargs="?",
        default="./data/AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5",
        help="Data graphs folder.",
    )

    parser.add_argument(
        "--model_folder", nargs="?", default="./saved_model", help="Save models"
    )

    parser.add_argument(
        "--prediction_path",
        nargs="?",
        default="./output",
        help="Path to store the predicted graph labels.",
    )

    parser.add_argument(
        "--log_path",
        nargs="?",
        default="./logs",
        help="Log json with parameters and performance.",
    )

    parser.add_argument(
        "--tensorboard_dir",
        nargs="?",
        default="./runs",
        help="Tensorboard log dir.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=45,
        help="Number of training epochs. Default is 45.",
    )

    parser.add_argument(
        "--node_number",
        type=int,
        default=90,
        help="The number of nodes depends on the input data. Default is 90.",
    )

    parser.add_argument(
        "--alpha",
        type=list,
        default=[0.01, 0.25, 0.5, 0.9, 0.99],
        help="The alpha vector for SPDSRU. Default is [0.01, 0.25, 0.5, 0.9, 0.99].",
    )

    parser.add_argument(
        "--class_number",
        type=int,
        default=2,
        help="class_number. Default is 8.",
    )

    parser.add_argument(
        "--dropout", type=float, default=0.25, help="Dropout rate. Default is 0.5."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of graphs processed per batch. Default is 32.",
    )

    parser.add_argument(
        "--time", type=int, default=100, help="Time budget for steps. Default is 100."
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=60,
        help="Number of predictive repetitions. Default is 60.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount for correct predictions. Default is 0.99.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate. Default is 0.001.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Learning rate. Default is 0.",
    )

    parser.add_argument(
        "--desc",
        nargs="?",
        default="",
        help="The description for the experiment.",
    )

    parser.add_argument(
        "--node_attention",
        action='store_true',
        default=False,
        help="Random walk by weighting all links using attention, and send the results that weight all node feature with attention to SPDSRU.",
    )

    parser.add_argument(
        "--reward",
        action='store_true',
        default=False,
        help="reward function: reward = 0 if target == prediction else reward = -1",
    )

    parser.add_argument(
        "--gnnexplainer_reward",
        action='store_true',
        default=False,
        help="the gnn explainer reward",
    )

    parser.add_argument(
        "--single_data",
        action='store_true',
        default=False,
        help="no sliding window",
    )

    parser.add_argument(
        "--task_hcp_data_two",
        action='store_true',
        default=False,
        help="task classification",
    )

    parser.add_argument(
        "--task_hcp_data_four",
        action='store_true',
        default=False,
        help="task classification",
    )

    return parser.parse_args()


class StepNetworkLayer(torch.nn.Module):
    """
    Step Network Layer Class for selecting next node to move.
    """

    def __init__(self, args):
        """
        Initializing the layer.
        :param args: Arguments object.
        :param identifiers: Node type -- id hash map.
        """
        super(StepNetworkLayer, self).__init__()
        self.args = args
        self.reset_attention()

    def forward(self, graph, node):
        """
        Sampling a node from the neighbourhood.
        :node: the current node
        :param graph: NetworkX graph.
        :return new node: a new node sampled from the neighbourhood with attention.
        """
        neighbor_weights = torch.tensor(
            [
                graph.get_edge_data(node, n)["weight"]
                if n in graph.neighbors(node)
                else 0.0
                for n in range(self.args.node_number)
            ]
        )

        if self.args.cuda:
            neighbor_weights = neighbor_weights.cuda()

        neighbor_weights = neighbor_weights * self.attention
        normalized_neighbor_weights = neighbor_weights / neighbor_weights.sum()
        normalized_neighbor_weights = (
            normalized_neighbor_weights.detach().cpu().numpy().reshape(-1)
        )
        new_node = np.random.choice(
            np.arange(self.args.node_number), p=normalized_neighbor_weights
        )
        attention_score = self.attention[new_node]
        return new_node, attention_score

    def reset_attention(self):
        self.attention = torch.ones(self.args.node_number) / self.args.node_number
        if self.args.cuda:
            self.attention.cuda()


class DownStreamNetworkLayer(torch.nn.Module):
    """
    Neural network layer for attention update and node label assignment.
    """

    def __init__(self, args):
        """
        :param args:
        :param target_number:
        :param identifiers:
        """
        super(DownStreamNetworkLayer, self).__init__()
        self.args = args
        self.create_parameters()

    def create_parameters(self):
        """
        Defining and initializing the classification and attention update weights.
        """
        self.attention_net = torch.nn.Sequential(
            SPDLogMap(),
            SPDVectorize(),
            torch.nn.Linear(
                self.args.wavelets_num**2,
                self.args.node_number,
            ),
            torch.nn.Softmax(),
        )

    def forward(self, hidden_state):
        """
        Making a forward propagation pass with the input from the LSTM layer.
        :param hidden_state: LSTM state used for labeling and attention update.
        """
        attention = self.attention_net(hidden_state)
        return attention


class GAM(torch.nn.Module):
    """
    Graph Attention Machine class.
    """

    def __init__(self, args):
        """
        Initializing the machine.
        :param args: Arguments object.
        """
        super(GAM, self).__init__()
        self.args = args

        self.step_block = StepNetworkLayer(self.args)
        self.recurrent_block = SPDSRU(
            self.args.alpha, self.args.batch_size, self.args.wavelets_num
        )
        self.down_block = DownStreamNetworkLayer(self.args)
        self.classify = torch.nn.Sequential(
            SPDLogMap(),
            SPDVectorize(),
            torch.nn.Linear(
                self.args.wavelets_num**2,
                self.args.class_number,
            ),
        )
        self.reset_attention()

    def forward(self, graph, features, node):
        """
        Doing a forward pass on a graph from a given node.
        :param data: Data dictionary.
        :param graph: NetworkX graph.
        :param features: Feature tensor.
        :param node: Source node identifier.
        :return label_predictions: Label prediction.
        :return node: New node to move to.
        :return attention_score: Attention score on selected node.
        """
        node, attention_score = self.step_block(graph, node)  # call step network forward
        if self.args.node_attention:
            node_feat = (
                self.step_block.attention.unsqueeze(-2) @ features.transpose(-3, -2)
            ).squeeze(-2)
        else:
            node_feat = features[node]
        lstm_output, self.state = self.recurrent_block(
            node_feat, self.state
        )  # TODO: 最开始的顶点特征没有进入网络
        if self.args.node_attention:
            self.step_block.attention = self.down_block(lstm_output).squeeze(0)
        logits = self.classify(lstm_output)
        return logits, node, attention_score, lstm_output

    def reset_attention(self):
        self.state = torch.diag_embed(
            1e-1
            * torch.ones(
                self.args.batch_size,
                len(self.args.alpha),
                self.args.wavelets_num,
                requires_grad=True,
            )
        )
        self.step_block.reset_attention()
        if self.args.cuda:
            self.state.cuda()


class Attention(torch.nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = torch.nn.Linear((enc_hid_dim * 2), dec_hid_dim)
        self.v = torch.nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.nn.functional.softmax(attention, dim=1)


class GAMTrainer(torch.nn.Module):
    """
    Object to train a GAM model.
    """

    def __init__(self, args):
        super(GAMTrainer, self).__init__()
        self.max_seqlen = 8

        self.args = args
        self.args.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.model = GAM(self.args)
        self.target_number = self.args.class_number
        self.seq_encoder = SPDLSTM(
            self.args.alpha, self.args.batch_size, self.args.wavelets_num
        )

        self.writer = SummaryWriter(
            log_dir=os.path.join(self.args.tensorboard_dir, self.args.current_time)
        )

        self.args.model_folder = os.path.join(
            self.args.model_folder, f'{self.args.current_time}'
        )
        os.makedirs(self.args.model_folder, exist_ok=True)

        self.init_log()
        self.setup_data()
        self.create_parameters()
        self.train_steps = []
        self.test_steps = []
        self.seq_attn = []

        self.args_printer()

    def init_log(self):
        self.args.log_path = os.path.join(
            self.args.log_path, f'{self.args.current_time}'
        )
        os.makedirs(self.args.log_path, exist_ok=True)

        root_logger = logging.getLogger()
        formatter = logging.Formatter(
            '%(asctime)s\t%(name)s\t%(pathname)s:%(lineno)d\t%(funcName)s\t%(levelname)s\t%(message)s'
        )
        file_handler = logging.FileHandler(
            os.path.join(self.args.log_path, 'console.log'), encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)
        logging.getLogger('trainer').addHandler(logging.StreamHandler())

        self.logs = {
            'train_losses': [],
            'train_accs': [],
            'params': vars(self.args),
            'model_time': self.args.current_time,
        }

    def args_printer(self):
        """
        Function to print the logs in a nice tabular format.
        :param args: Parameters used for the model.
        """
        args = vars(self.args)
        keys = sorted(args.keys())
        t = Texttable()
        t.add_rows([["Parameter", "Value"]] + [[k, args[k]] for k in keys])
        logging.getLogger('trainer').info('\n' + t.draw())

    def setup_data(self):
        """
        Listing the training and testing graphs in the source folders.
        """
        # dataset = GraphSeqDataset(
        #     path=self.args.data_folder,
        #     window=self.args.data_window,
        #     step=self.args.data_step,
        #     padding=self.args.data_padding,
        #     ratio=self.args.thresholding_ratio,
        #     wavelets_num=self.args.wavelets_num,
        #     beta=self.args.wavelets_beta,
        #     gamma=self.args.wavelets_gamma,
        #     max_iter=self.args.wavelets_max_iter,
        #     min_err=self.args.wavelets_min_err,
        #     node_select=self.args.wavelets_node_select,
        # )

        dataset = GraphSeqFileDataset(path=self.args.data_folder)
        train_num = int(len(dataset) * 0.75)
        test_num = len(dataset) - train_num
        self.train_dataset, self.test_dataset = random_split(
            dataset,
            [train_num, test_num],
            generator=torch.Generator().manual_seed(self.args.seed),
        )

    def create_parameters(self):
        """
        Defining and initializing the classification and attention update weights.
        """

        self.classify = torch.nn.Sequential(
            SPDLogMap(),
            SPDVectorize(),
            torch.nn.Linear(
                self.args.wavelets_num**2,
                self.args.class_number,
            ),
        )

    def process_graph(self, graph, features, target, o_filenames, epoch):
        """
        Reading a graph and doing a forward pass on a graph with a time budget.
        :param graph_path: Location of the graph to process.
        :param _loss: Loss on the graphs processed so far in the batch.
        :return _loss: Incremented loss on the current batch being processed.
        """
        if self.args.cuda:
            features = features.cuda()
            target = target.cuda()
        node = random.choice(list(graph.nodes()))  # randomly choose a node to start
        start_node = node
        steps = []
        gam_output = []
        loss = 0
        attention_loss = 0
        gnn_reward = 0
        for t in range(self.args.time):
            logits, node, attention_score, single_gam_output = self.model(
                graph, features, node
            )  # call GAM forward
            steps.append(str(node))
            gam_output.append(single_gam_output)
            prediction_loss = torch.nn.functional.cross_entropy(logits, target)
            loss += prediction_loss
            # gnn_reward -= torch.argmax(logits)*torch.log(torch.argmax(logits))
            if t < self.args.time-2:
                # attention fading through steps, first step has least importance 0.99^10
                # log(attention_score) will always be negative as attention_score < 1
                assert 0 <= attention_score and attention_score <= 1, "attention score out of range [0,1]"
                attention_loss += (self.args.gamma ** (self.args.time - t)) * torch.log(attention_score)

        reward = calculate_reward(target, logits)  # reward = 0 if target == prediction else reward = -1
        gnn_reward = calculate_gnnexplainer_reward(target, logits)

        if self.args.gnnexplainer_reward:
            loss = loss + gnn_reward*attention_loss
        else:
            loss = loss + reward*attention_loss  # correct prediction will decrease loss

        # save steps
        if self.args.debug:
            train_steps = {
                "start_node": start_node,
                "train_steps": steps,
                "attention": list(self.model.step_block.attention.detach().cpu().numpy().astype(str)),
                "attention_loss": attention_loss.detach().cpu().numpy().reshape(-1).astype(str)[0],
                "loss": loss.detach().cpu().numpy().reshape(-1).astype(str)[0],
                "reward": reward.detach().cpu().numpy().reshape(-1).astype(str)[0],
                "gam_output": gam_output,
                "target": target,
                "logits": logits,
            }
            # train_steps
            train_steps_file = (
                self.args.prediction_path
                + f"/trainsteps_{os.path.split(self.args.data_folder)[1]}/{self.args.epochs}_{self.args.repetitions}_{self.args.time}"
            )# {self.args.current_time}_{self.args.mode}_{o_filenames}_{epoch}
            save_data_singledata(train_steps, train_steps_file, o_filenames, epoch)
        self.model.reset_attention()
        return loss, single_gam_output, target, steps

    def process_batch(self, batch):
        """
        Forward and backward propagation on a batch of graphs.
        :param batch: Batch if graphs.
        :return loss_value: Value of loss on batch.
        """
        self.optimizer.zero_grad()  # Clears the gradients of all optimized torch.Tensor s.
        batch_loss = 0

        if self.args.mode in ["all", "firstscan"]:
            for b in batch:
                graph_path = self.args.data_folder + b + ".json"
                batch_loss, _, _, _ = self.process_graph(graph_path, batch_loss)
        elif self.args.mode == "seq":
            print("TODO")
        else:
            print("WRONG mode!")

        batch_loss.backward(retain_graph=True)
        self.optimizer.step()
        loss_value = batch_loss.item()
        self.optimizer.zero_grad()
        return loss_value

    def seq_forward(self, seq, epoch):
        seq_inputs = []
        loss = 0
        if self.args.single_data:
            g, features, target, o_filenames = seq  # _, _, _,
            # print(type(seq))
            # print(g)
            # print(features)
            # print(target)
            # print(o_filenames)
            features = features[0].reshape(
                -1, self.args.wavelets_num, self.args.wavelets_num
            )
            target = target.reshape(-1)
            # print(target)
            if args.task_hcp_data_two:
                if int(target) in [1, 3, 4, 6]:
                    target = torch.tensor([0])
                else:
                    target = torch.tensor([1])
            if args.task_hcp_data_four:
                if int(target) in [0, 3]:
                    target = torch.tensor([0])
                elif int(target) in [1, 5]:
                    target = torch.tensor([1])
                elif int(target) in [2, 6]:
                    target = torch.tensor([2])
                else:
                    target = torch.tensor([3])
            graph_loss, graph_output, target, steps = self.process_graph(g[0], features, target, o_filenames, epoch)
            seq_inputs.append(
                graph_output.reshape(-1, self.args.wavelets_num, self.args.wavelets_num)
            )
            loss += graph_loss

            enc_out = seq_inputs
        else:
            _, _, _, o_filenames = seq
            # o_filenames = seq[3]
            for g, features, target in zip(*seq[0:3]):  # *seq # , o_filenames
                features = features.reshape(
                    -1, self.args.wavelets_num, self.args.wavelets_num
                )
                target = target.reshape(-1)
                # print(target)
                if args.task_hcp_data_two:
                    if int(target) in [1, 3, 4, 6]:
                        target = torch.tensor([0])
                    else:
                        target = torch.tensor([1])
                if args.task_hcp_data_four:
                    if int(target) in [0, 3]:
                        target = torch.tensor([0])
                    elif int(target) in [1, 5]:
                        target = torch.tensor([1])
                    elif int(target) in [2, 6]:
                        target = torch.tensor([2])
                    else:
                        target = torch.tensor([3])
                graph_loss, graph_output, target, steps = self.process_graph(g, features, target, o_filenames, epoch)
                seq_inputs.append(
                    graph_output.reshape(-1, self.args.wavelets_num, self.args.wavelets_num)
                )
                loss += graph_loss

            seq_in = torch.stack(seq_inputs)  # (seq_len, batch, wavelets_num, wavelets_num)
            # seq encoding
            enc_out, hidden = self.seq_encoder(
                seq_in
            )  # enc_out (src_len, batch, enc hid dim)
        # hidden in LSTM is a tuple of (h_n, c_n)
        logits = self.classify(enc_out[-1])
        pred_loss = torch.nn.functional.cross_entropy(logits, target)

        if self.args.reward:
            pred_loss = pred_loss + loss

        if self.args.debug:
            self.seq_attn.append(
                {
                    "graph_seq": list(seq)
                    # "seq_in": list(seq_inputs)
                }
            )
        return pred_loss, logits, target, enc_out, steps

    def process_seq(self, seq, epoch):
        """
        Forward and backward propagation on a batch of graphs.
        :param Seq: sequence of graphs [g1,g2,g3, ...] with target
        :return loss_value: Value of loss on seq.
        """
        seq_loss, logits, target, enc_out, _ = self.seq_forward(seq, epoch)
        seq_loss.backward()
        self.optimizer.step()
        loss_value = seq_loss.item()
        self.optimizer.zero_grad()

        return loss_value, logits, target, enc_out

    def update_log(self):
        """
        Adding the end of epoch loss to the log.
        """
        if self.args.mode == "seq":
            average_loss = self.loss_meter.avg
        else:
            average_loss = self.loss_meter.avg
        self.logs["train_losses"].append(average_loss)
        self.logs["train_accs"].append(self.acc_meter.avg)

    def fit(self):
        """
        Fitting a model on the training dataset.
        """
        logging.getLogger('trainer').info("\nTraining started.\n")
        self.model.train()  # Sets the module in training mode.
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.optimizer.zero_grad()  # Clears the gradients of all optimized torch.Tensor s.
        # start training
        best_epoch = 0
        patience = 0
        min_loss = inf
        step = 1
        logging.getLogger().info(f'epoch\tloss\tacc')
        for epoch in range(1, self.args.epochs + 1):
            random.shuffle(self.train_dataset.indices)
            print(f'\nepoch {epoch}')

            self.acc_meter = AverageMeter()
            self.loss_meter = AverageMeter()

            if self.args.mode == "seq":
                bar = tqdm(self.train_dataset)
                for seq in bar:
                    seq_loss, logits, target, enc_out = self.process_seq(seq, epoch)
                    self.acc_meter.update(
                        torch.sum(logits.argmax(-1) == target).item(), len(logits)
                    )
                    self.loss_meter.update(seq_loss)
                    self.writer.add_scalar('step_loss/train', seq_loss, step)
                    step += 1
                    bar.set_description(
                        f'seq_loss: {seq_loss:.4f} | loss: {self.loss_meter.avg:.4f} | acc: {self.acc_meter.avg:.4f}'
                    )

            elif self.args.mode in ["all", "firstscan"]:
                batches = create_batches(self.train_x, self.args.batch_size)
                self.epoch_loss = 0
                self.nodes_processed = 0
                for batch in range(len(batches)):
                    self.epoch_loss = self.epoch_loss + self.process_batch(
                        batches[batch]
                    )
                    self.nodes_processed = self.nodes_processed + len(batches[batch])
                    loss_score = round(self.epoch_loss / self.nodes_processed, 4)
            else:
                logging.getLogger('trainer').error("WRONG mode!")

            self.writer.add_scalar('epoch_loss/train', self.loss_meter.avg, epoch)
            self.writer.add_scalar('epoch_acc/train', self.acc_meter.avg, epoch)
            logging.getLogger().info(
                f'{epoch}\t{self.loss_meter.avg}\t{self.acc_meter.avg}'
            )
            self.update_log()
            # save best model
            if self.loss_meter.avg < min_loss:
                best_epoch = epoch
                min_loss = self.loss_meter.avg
                patience = 0
                save_model_path = os.path.join(
                    self.args.model_folder,
                    f"model_epoch-{epoch}_loss-{min_loss:.4f}_acc-{self.acc_meter.avg:.4f}.pt",
                )
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': min_loss,
                        'args': self.args,
                    },
                    save_model_path,
                )
            else:
                patience += 1
            # early stop
            if patience == self.args.early_stop:
                self.logs["earlystop_epoch"] = epoch
                save_model_path = os.path.join(
                    self.args.model_folder,
                    f"model_epoch-{epoch}_loss-{min_loss:.4f}_acc-{self.acc_meter.avg:.4f}.pt",
                )
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.logs["train_losses"][-1],
                        'args': self.args,
                    },
                    save_model_path,
                )
                break

            # if epoch % 5 == 0 and epoch > 20:
            #     model.score()

    def score_graph(self, target, prediction):
        """
        Scoring the prediction on the graph.
        :param data: Data hash table of graph.
        :param prediction: Label prediction.
        """
        target = target.item()
        is_it_right = target == prediction
        self.raw_pred.append(prediction)
        self.predictions.append(is_it_right)

    def score(self, data=None):
        """
        Scoring the test set graphs.
        """
        logging.getLogger('trainer').info("\nScoring the test set.\n")
        if data is None:
            data = self.test_dataset
        self.model.eval()
        self.raw_pred = []
        self.predictions = []
        test_loss_meter = AverageMeter()
        for b in tqdm(data):
            if self.args.mode == "seq":
                graph_predictions = []
                loss_meter = AverageMeter()
                for repet in range(self.args.repetitions):
                    loss, predictions, target, enc_out, steps = self.seq_forward(b, repet)
                    graph_predictions.append(
                        np.argmax(predictions.detach().cpu().numpy())
                    )
                    loss_meter.update(loss.item())
                    if self.args.debug:
                        # save steps
                        _, _, _, o_filenames = b
                        test_steps = {
                            "graph": b,
                            "repet": repet,
                            "attention": list(self.model.step_block.attention.detach().cpu().numpy().astype(str)),
                            "trest_steps": steps,
                            "enc_out": enc_out,
                            "pred_output": graph_predictions,
                            "target": target,
                        }
                        # test_steps
                        test_steps_file = (
                            self.args.prediction_path
                            + f"/teststeps_{os.path.split(self.args.data_folder)[1]}/{self.args.epochs}_{self.args.repetitions}_{self.args.time}"
                        )# {self.args.current_time}_{self.args.mode}_{o_filenames}_{repet}
                        save_data_singledata(test_steps, test_steps_file, o_filenames, repet)
            else:
                graph_path = self.args.data_folder + b + ".json"
                data = json.loads(codecs.open(graph_path, "r", encoding='utf-8').read())
                graph, features, target = load_data(
                    data
                )  # nx.Graph, torch.tensor, intiger
                if self.args.cuda:
                    features = features.cuda()
                    target = target.cuda()

                graph_predictions = []
                for repet in range(self.args.repetitions):
                    node = random.choice(list(graph.nodes()))
                    start_node = node
                    steps = []
                    for _ in range(self.args.time):
                        prediction, node, _, gam_output = self.model(graph, features, node)
                        steps.append(str(node))
                    graph_predictions.append(
                        np.argmax(prediction.detach().cpu().numpy())
                    )
                    # if self.args.debug:
                    #     # save steps
                    #     self.test_steps.append(
                    #         {
                    #             "graph": b,
                    #             "repet": repet,
                    #             "attention": list(self.model.step_block.attention.detach().cpu().numpy().astype(str)),
                    #             "start_node": str(start_node),
                    #             "trest_steps": steps,
                    #             "gam_output": gam_output.detach().cpu().numpy().reshape(-1).astype(str)[0],
                    #             "pred_output": graph_predictions.detach().cpu().numpy(),
                    #         }
                    #     )
                    self.model.reset_attention()
            prediction = max(set(graph_predictions), key=graph_predictions.count)
            test_loss_meter.update(loss_meter.avg)
            self.score_graph(target, prediction)
            # self.conf_mat = confusion_matrix(target, prediction)
        self.accuracy = float(np.mean(self.predictions))
        self.logs["test_loss"] = test_loss_meter.avg
        self.writer.add_scalar('acc/test', self.accuracy, 0)
        self.writer.add_scalar('loss/test', test_loss_meter.avg, 0)
        hparams = {
            k: (v if type(v) != list else ','.join(map(str, v)))
            for k, v in vars(self.args).items()
        }
        metrics = {
            'hparam/train_loss': self.logs["train_losses"][-1],
            'hparam/train_acc': self.logs["train_accs"][-1],
            'hparam/test_loss': test_loss_meter.avg,
            'hparam/test_acc': self.accuracy,
        }
        self.writer.add_hparams(hparams, metrics)
        logging.getLogger('trainer').info(
            f"\nThe test set loss is: {test_loss_meter.avg:.4f}"
        )
        logging.getLogger('trainer').info(
            f"The test set accuracy is: {self.accuracy:.4f}"
        )

    def save_predictions_and_logs(self):
        """
        Saving the predictions as a csv file and logs as a JSON.
        """
        # logs
        self.logs["test_acc"] = self.accuracy
        # self.logs["pred_label"] = self.raw_pred
        # self.logs["confusion matrix"] = self.conf_mat
        os.makedirs(self.args.log_path, exist_ok=True)
        log_file = os.path.join(self.args.log_path, 'data_log.json')
        with open(log_file, "w", encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False)

        cols = ["graph_id", "predicted_label"]
        predictions = [
            [self.test_dataset[i], self.predictions[i].item()]
            for i in range(len(self.test_dataset))
        ]
        # output
        self.output_data = pd.DataFrame(predictions, columns=cols)
        output_file = (
            self.args.prediction_path
            + f"/predictions_{self.args.current_time}_{self.args.mode}.csv"
        )
        self.output_data.to_csv(output_file, index=None)

        if not self.args.debug:
            # train_steps
            train_steps_file = (
                self.args.prediction_path
                + f"/trainsteps_{self.args.current_time}_{self.args.mode}"
            )
            save_data_singledata(self.train_steps, train_steps_file)
            # test_steps
            test_steps_file = (
                self.args.prediction_path
                + f"/teststeps_{self.args.current_time}_{self.args.mode}"
            )
            save_data_singledata(self.test_steps, test_steps_file)
            # json.dump(
            #     self.train_steps,
            #     codecs.open(train_steps_file, "w", encoding='utf-8'),
            #     separators=(',', ':'),
            #     sort_keys=False,
            #     indent=4,
            # )
            # json.dump(
            #     self.test_steps,
            #     codecs.open(test_steps_file, "w", encoding='utf-8'),
            #     separators=(',', ':'),
            #     sort_keys=False,
            #     indent=4,
            # )
            # seq_attn
            # seq_attn_file = (
            #     self.args.model_folder
            #     + f"seq_attn_{self.args.current_time}_{self.args.mode}.json"
            # )
            # json.dump(
            #     self.seq_attn,
            #     codecs.open(seq_attn_file, "w", encoding='utf-8'),
            #     separators=(',', ':'),
            #     sort_keys=False,
            #     indent=4,
            # )


# Start training
"""
Parsing command line parameters, processing graphs, fitting a GAM.
"""
if __name__ == '__main__':
    args = parameter_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = GAMTrainer(args)
    if args.cuda:
        model = model.cuda()

    model.fit()
    model.score()
    model.save_predictions_and_logs()
