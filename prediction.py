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
from texttable import Texttable

from torch.nn.functional import cross_entropy
from sklearn.metrics import confusion_matrix
from train import GAMTrainer, parameter_parser

args = parameter_parser()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
model = GAMTrainer(args)
checkpoint = torch.load("D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\IPMI2023\\IPMI-2023\\saved_model\\2022-10-27_16-08-38\\model_epoch-40_loss-1211.4964_acc-0.9474.pt")
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.score()
model.save_predictions_and_logs()
