from logging import exception
import numpy as np
from scipy.linalg import expm

# from scipy.sparse.linalg import norm # not sparse matrix
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from timeit import timeit
from utils import cov, corrcoef
import warnings

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def thresholding(fc, data_path, ratio=0.8):
    # keeping T*100% links
    node_num = fc.shape[-1]
    fc[fc < 0] = 0

    fc_tril = np.tril(fc, -1)
    K = np.count_nonzero(fc_tril)
    KT = ratio * ((node_num**2 - node_num) / 2)
    KT = math.ceil(KT)

    if KT >= K:
        thr = 0
    else:
        thr = np.partition(fc_tril.reshape(*fc_tril.shape[:-2], -1), -KT, -1)[
            ..., [[-KT]]
        ]

    fc[fc < thr] = 0

    if not np.all(np.sum(fc > 0, -1) > 1):
        print('the wrong datafile: ', data_path)
        warnings.warn('Thresholding is too large, please enlarge the threshold')

    return fc


def harmonic_wavelets(
    graph,
    wavelets_num=10,
    beta=1,
    gamma=0.005,
    max_iter=1000,
    min_err=0.0001,
    node_select=10,
):
    # # Indiviual wavelets
    # node_num = graph.shape[-1]
    # temp_D = np.eye(node_num) * np.sum(graph, axis=-1, keepdims=True)
    # latentlaplacian = temp_D - graph
    # u_vec = np.zeros_like(graph)
    #
    # diag_idx = np.arange(node_num)
    # np.put_along_axis(
    #     u_vec, np.argpartition(graph, -node_select)[..., -node_select:], 1, -1
    # )
    # u_vec[..., diag_idx, diag_idx] = 1
    #
    # temp_v = 1 - u_vec
    # temp_v = np.eye(temp_v.shape[-1]) * np.expand_dims(temp_v, -2)
    # Theta = beta * temp_v
    # _, temp_phi = np.linalg.eigh(latentlaplacian + Theta)
    # phi_k = np.expand_dims(temp_phi[..., :wavelets_num], -3)

    # Common wavelets, still a little error in objective function
    node_num = graph.shape[-1]
    temp_D = np.eye(node_num) * np.sum(graph, axis=-1, keepdims=True)
    latentlaplacian = temp_D - graph
    _, temp_phi = np.linalg.eigh(latentlaplacian)
    u_vec = np.zeros_like(graph)
    phi_k = np.expand_dims(temp_phi[..., :wavelets_num], -3)

    it = 0
    err = np.inf
    diag_idx = np.arange(node_num)
    while err > min_err and it < max_iter:
        np.put_along_axis(
            u_vec, np.argpartition(graph, -node_select)[..., -node_select:], 1, -1
        )
        u_vec[..., diag_idx, diag_idx] = 1

        temp_v = 1 - u_vec
        temp_v = np.eye(temp_v.shape[-1]) * np.expand_dims(temp_v, -2)
        Theta = beta * temp_v
        temp_increment = 2 * (np.eye(node_num) - phi_k @ phi_k.swapaxes(-2, -1))
        phi_increment = -gamma * temp_increment @ Theta @ phi_k
        Q, R = np.linalg.qr(
            (np.eye(node_num) - phi_k @ phi_k.swapaxes(-2, -1)) @ phi_increment
        )
        A = phi_k.swapaxes(-2, -1) @ phi_increment
        temp_matrix1 = np.concatenate([A, -R.swapaxes(-2, -1)], -1)
        temp_matrix2 = np.concatenate([R, np.zeros_like(R)], -1)
        temp_matrix3 = np.concatenate([temp_matrix1, temp_matrix2], -2)
        BC = expm(temp_matrix3)[..., :wavelets_num]
        phi_k = phi_k @ BC[..., :wavelets_num, :] + Q @ BC[..., wavelets_num:, :]
        err = np.max(np.linalg.norm(phi_increment, 'fro', (-2, -1)))
        it += 1
        print(it, err)

    return phi_k


def harmonics(
    graph,
    wavelets_num=55,
    beta=1,
    gamma=0.005,
    max_iter=1000,
    min_err=0.0001,
):
    node_num = graph.shape[-1]
    temp_D = np.eye(node_num) * np.sum(graph, axis=-1, keepdims=True)
    latentlaplacian = temp_D - graph
    _, temp_phi = np.linalg.eigh(latentlaplacian)
    phi_k = np.expand_dims(temp_phi[..., :wavelets_num], -3)

    # it = 0
    # err = np.inf
    # while err > min_err and it < max_iter:
    #     temp_increment = 2 * (np.eye(node_num) - phi_k @ phi_k.swapaxes(-2, -1))
    #     phi_increment = -gamma * temp_increment @ phi_k
    #     Q, R = np.linalg.qr(
    #         (np.eye(node_num) - phi_k @ phi_k.swapaxes(-2, -1)) @ phi_increment
    #     )
    #     A = phi_k.swapaxes(-2, -1) @ phi_increment
    #     temp_matrix1 = np.concatenate([A, -R.swapaxes(-2, -1)], -1)
    #     temp_matrix2 = np.concatenate([R, np.zeros_like(R)], -1)
    #     temp_matrix3 = np.concatenate([temp_matrix1, temp_matrix2], -2)
    #     BC = expm(temp_matrix3)[..., :wavelets_num]
    #     phi_k = phi_k @ BC[..., :wavelets_num, :] + Q @ BC[..., wavelets_num:, :]
    #     err = np.max(np.linalg.norm(phi_increment, 'fro', (-2, -1)))
    #     it += 1
    #     print(it, err)

    return phi_k


def cfc(wavelets, bold, wavelets_num=10):
    # a = np.expand_dims(wavelets[..., :wavelets_num].swapaxes(-2, -1), -3)
    # b = np.expand_dims(np.expand_dims(bold.swapaxes(-2, -1), -2), -4)
    # c = a*b
    powers = np.sum(
        np.expand_dims(wavelets[..., :wavelets_num].swapaxes(-2, -1), -3)
        * np.expand_dims(np.expand_dims(bold.swapaxes(-2, -1), -2), -4),
        -1,
    )
    cfcs = corrcoef(powers.swapaxes(-2, -1))
    return cfcs, powers


def get_cfc(bold, data_path):
    fc = np.corrcoef(bold)
    Graph = thresholding(fc, data_path)
    wavelets = harmonic_wavelets(Graph)
    # wavelets = harmonics(Graph)
    cfc_matrix, power = cfc(wavelets, BOLD_window)
    return cfc_matrix


if __name__ == '__main__':
    # BOLD_window is the bold signal data of a single subject, each row stands for the signal of each brain node
    # BOLD_window = np.loadtxt('data/TimeSeries101.csv', delimiter=',')
    np.random.seed(111)
    BOLD_window = np.random.rand(90, 100)
    # df = np.transpose(BOLD_window)
    # df = pd.DataFrame(df)
    # fc = df.corr()
    # fc = fc.values
    # # Graph is the adjacence matrix of the brain network(the topology of network)
    # Graph = thresholding(fc)
    # wavelets = harmonic_wavelets(Graph)
    # # cfc_matrix is the CFC matrix(SPD matrix)
    # cfc_matrix, power = cfc(wavelets, BOLD_window)
    # print(cfc_matrix)
    a = get_cfc(BOLD_window[..., :15], data_path='')
    b = get_cfc(BOLD_window[..., 15:30], data_path='')
    print(a)
    print(b)

# # fig1 = plt.figure()
# plt.matshow(Graph)
# plt.title('Graph')
# plt.savefig('./results/Graph.svg', format='svg', dpi=500)
#
# wavelets_num = 10
# node_num = len(Graph)
# wavelets_all = np.zeros((node_num, node_num*wavelets_num))
# for i in range(0, node_num):
#     wavelets_all[:, wavelets_num*i:wavelets_num*(i+1)] = wavelets[i]
# # fig2 = plt.figure()
# plt.matshow(wavelets_all)
# plt.title('wavelets')
# plt.savefig('./results/wavelets.svg', format='svg', dpi=500)
#
# node_num = len(Graph)
# colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'lime', 'pink')
# for N_i in range(0, node_num):
#     x = np.arange(0, 268, step=1)
#     plt.figure(figsize=(8.5, 5.5))
#     # fig3 = plt.figure()
#     for i in range(0, 10):
#         # _range = np.max(power[N_i,:,i,0]) - np.min(power[N_i,:,i,0])
#         # power[N_i,:,i,0] = (power[N_i,:,i,0] - np.min(power[N_i,:,i,0]))/_range
#         plt.plot(power[N_i, :, i, 0], color=colors[i], label='%s' % (i + 1))
#     plt.legend()
#     plt.title('power')
#     # plt.gcf().set_size_inches(18.5, 10.5)
#     plt.savefig('./results/power_node{}.svg'.format((N_i + 1)), format='svg', dpi=500)
#
#     x = np.arange(0, 268, step=1)
#     # fig3 = plt.figure()
#     plt.figure(figsize=(10.5, 18.5))
#     for i in range(0, 10):
#         # _range = np.max(power[N_i,:,i,0]) - np.min(power[N_i,:,i,0])
#         # power[N_i,:,i,0] = (power[N_i,:,i,0] - np.min(power[N_i,:,i,0]))/_range
#         plt.subplot(5,2,i+1)
#         plt.plot(power[N_i,:,i,0], color=colors[i])
#         plt.title('%s'%(i+1))
#     # plt.gcf().set_size_inches(18.5, 10.5)
#     plt.savefig('./results/sec_power_node{}.svg'.format((N_i+1)), format='svg', dpi=500)
#     plt.pause(2)
#     plt.close('all')
#
#     for i in range(0, 10):
#         cfc_matrix[N_i,i,i] = 0
#     # _range = np.max(cfc_matrix[N_i]) - np.min(cfc_matrix[N_i])
#     # cfc_matrix[N_i] = (cfc_matrix[N_i] - np.min(cfc_matrix[N_i])) / _range
#     # fig4 = plt.figure()
#     ax = plt.matshow(cfc_matrix[N_i])
#     plt.colorbar(ax.colorbar, fraction=0.025)
#     plt.title('cfc_matrix')
#     plt.savefig('./results/cfc_node{}.svg'.format((N_i+1)), format='svg', dpi=500)
#
# Graph_reduce = np.zeros((node_num, 10, 10))
# u_vec = np.zeros((node_num, node_num))
# node_inf = np.loadtxt('node_all.txt', delimiter=' ', usecols=range(7))
# node_reduce = np.zeros((node_num, 10, 6))
# for N_i in range(0, node_num):
#     Graph_col = Graph[:, N_i]
#     val_graph_id = sorted(range(len(Graph_col)), key=lambda k: Graph_col[k], reverse=True)
#     for i in range(0, 10):
#         u_vec[N_i, val_graph_id[i]] = 1
#     u_vec[N_i, N_i] = 1
#     k = 0
#     l = 0
#     node_inf_temp = node_inf[:, 0:6]
#     Graph_reduce_temp = Graph
#     for j in range(0, node_num):
#         if u_vec[N_i, j] == 0:
#             Graph_reduce_temp = np.delete(Graph_reduce_temp, j-l, axis = 0)
#             Graph_reduce_temp = np.delete(Graph_reduce_temp, j-l, axis = 1)
#             l = l + 1
#         elif u_vec[N_i, j] == 1 and N_i == j:
#             node_reduce[N_i, k, :] = node_inf_temp[j, 0:6]/2
#             node_reduce[N_i, k, 4] = 1
#             k = k + 1
#         else:
#             node_reduce[N_i, k, :] = node_inf_temp[j, 0:6]/2
#             k = k + 1
#     Graph_reduce[N_i] = Graph_reduce_temp
#     np.savetxt('./brain_picture/edge{}.txt'.format((N_i + 1)), Graph_reduce[N_i], fmt='%s', delimiter='\t')
#     np.savetxt('./brain_picture/node{}.txt'.format((N_i + 1)), node_reduce[N_i], fmt='%s', delimiter='\t')
#
# node_reduce = np.zeros((node_num, 10, 10, 6))
# N_i = 211
# node_inf_temp = node_inf[:, 0:6]
# for m in range(0, 10):
#     k = 0
#     l = 0
#     for j in range(0, node_num):
#         if u_vec[N_i, j] == 0:
#             l = l + 1
#         else:
#             node_reduce[N_i, m, k, :] = node_inf_temp[j, 0:6]/2
#             node_reduce[N_i, m, k, 5] = wavelets[N_i, j, m]
#             k = k + 1
#     np.savetxt('./brain_picture/node{}_{}.txt'.format((N_i + 1), m), node_reduce[N_i, m], fmt='%s', delimiter='\t')
