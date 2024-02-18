import torch
from torch import nn


class SPDVectorize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input.flatten(-2, -1)
        return output


class SPDLogMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        s, u = input.symeig(eigenvectors=True)
        s = s.log().diag_embed()
        output = u @ s @ u.transpose(-2, -1)
        return output


class SPDExpMap(nn.Module):
    def __init__(self):
        super(SPDExpMap, self).__init__()

    def forward(self, input):
        s, u = input.symeig(eigenvectors=True)
        s = s.exp().diag_embed()
        output = u @ s @ u.transpose(-2, -1)
        return output
