import math
import random
# from wavelets_test import plot_matfig, mat_to_ones
import re
from scipy.linalg import fractional_matrix_power
import joblib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

# plt.rc('font', family='Times New Roman', size=18, weight='bold')
import os
import pickle

from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
from utils import cov, corrcoef

from sklearn.metrics import confusion_matrix



def parameter_parser():
    parser = argparse.ArgumentParser(description="Plot CFC Pattern.")

    parser.add_argument(
        '--processed_data_path',
        nargs="?",
        default="./data/AAL_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5",
        help='The processed data in python format',
    )
    # ./data/AAL_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5, ./data/aal_cn_ad, ./data/hcp

    parser.add_argument(
        '--cfc_data_path',
        nargs="?",
        default="./data/AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5",
        help='The calculated cfc data',
    )
    # ./data/AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5, ./data/aal_cn_ad_cfc_iter1000, ./data/hcp_cfc

    parser.add_argument(
        "--box_plot",
        action='store_true',
        default=False,
        help="process the aal data",
    )

    parser.add_argument(
        "--aal",
        action='store_true',
        default=False,
        help="process the aal data",
    )

    parser.add_argument(
        "--hcp",
        action='store_true',
        default=False,
        help="process the aal data",
    )

    parser.add_argument(
        "--ocd",
        action='store_true',
        default=False,
        help="process the aal data",
    )

    parser.add_argument(
        "--task",
        action='store_true',
        default=False,
        help="process the aal data",
    )

    parser.add_argument(
        "--aal_pattern",
        action='store_true',
        default=False,
        help="process the aal data",
    )

    parser.add_argument(
        "--hcp_pattern",
        action='store_true',
        default=False,
        help="process the aal data",
    )

    parser.add_argument(
        "--ocd_pattern",
        action='store_true',
        default=False,
        help="process the aal data",
    )

    parser.add_argument(
        "--task_pattern",
        action='store_true',
        default=False,
        help="process the aal data",
    )

    parser.add_argument(
        "--training_pattern",
        action='store_true',
        default=False,
        help="process the aal data",
    )

    parser.add_argument(
        "--riemannian",
        action='store_true',
        default=False,
        help="process the aal data",
    )


    parser.add_argument(
        '--training_data_path',
        nargs="?",
        default="./output",# /trainsteps_2022-11-06_13-31-48_seq/result.pkl
        help='The processed data in python format',
    )

    parser.add_argument(
        '--testing_data_path',
        nargs="?",
        default="./output",# /teststeps_2022-11-06_13-31-48_seq/result.pkl
        help='The calculated cfc data',
    )

    parser.add_argument(
        '--o_data_path',
        nargs="?",
        default="./30_60_100", # trainsteps_aal_cn_ad_cfc_iter1000 # /trainsteps_2022-11-06_13-31-48_seq/result.pkl
        help='The processed data in python format',
    )

    parser.add_argument(
        '--to_data_path',
        nargs="?",
        default="./CFC matrix/25_30_120", # teststeps_aal_cn_ad_cfc_iter1000 # /teststeps_2022-11-06_13-31-48_seq/result.pkl
        help='The calculated cfc data',
    )

    parser.add_argument(
        "--node_num",
        type=int,
        default=90,
        help="Number of node. aal: 90; hcp: 116; ocd: 90; task: 268;",
    )

    parser.add_argument(
        "--wavelets_num",
        type=int,
        default=10,
        help="Number of wavelets.",
    )

    return parser.parse_args()


def fast_list2arr(data, offset=None, dtype=None):
    """
    Convert a list of numpy arrays with the same size to a large numpy array.
    This is way more efficient than directly using numpy.array()
    See
        https://github.com/obspy/obspy/wiki/Known-Python-Issues
    :param data: [numpy.array]
    :param offset: array to be subtracted from the each array.
    :param dtype: data type
    :return: numpy.array
    """
    num = len(data)
    out_data = np.empty((num,)+data[0].shape, dtype=dtype if dtype else data[0].dtype)
    for i in range(num):
        out_data[i] = data[i] - offset if offset else data[i]
    return out_data


def isValidIndex(x, n):
    return (x >= 0 and x < n)
    # 每一行的每个值的数组下标的差都一样，


def get_diag_value(cfc):
    sample_num = len(cfc)
    rows = cols = len(cfc[0])
    tem_arr = np.zeros((sample_num, 9, 9))  # 用来记录数组值
    for sample in range(sample_num):
        for i in range(0, cols-1):  # 共输出 cols * 2 - 1 行
            diff = cols - i - 1  # 每一行的差
            for j in range(cols):  # 数组中每一个值的下标范围是0到cols
                k = j - diff  # 通过一个下标值计算另一个下标值
                if isValidIndex(k, rows):  # 剩下就是判断这些下标值是否满足当前的情况， 这一步不怎么好理解
                    tem_arr[sample, i, k] = cfc[sample, k, j]

    tem_var = np.zeros((9, sample_num))
    for i in range(9):
        for sample_i in range(sample_num):
            tem_var[i, sample_i] = np.var(tem_arr[sample_i, i][tem_arr[sample_i, i]!=0])# , axis = 1

    tem_icc = np.zeros((9, 100))
    for i in range(6, 9):
        for loop in range(100):
            print("line: {}; loop: {}".format(i,loop))
            select_list = random.sample(range(sample_num), int(0.2 * sample_num))
            select_data = np.zeros((int(0.2*sample_num), i+1))
            count_select_num = 0
            for sample_s in select_list:
                # print(tem_arr[sample_s,i])
                select_data[count_select_num, :] = tem_arr[sample_s,i][tem_arr[sample_s,i]!=0]
                count_select_num = count_select_num + 1
            icc_type = "icc(1)"
            _, tem_icc[i,loop] = icc_calculate(select_data, icc_type)

    return tem_var, tem_icc


def get_diag_mean(cfc):
    sample_num = len(cfc)
    rows = cols = len(cfc[0])
    tem_arr = np.zeros((sample_num, 9, 9))  # 用来记录数组值
    tem_mean = np.zeros((9, sample_num))
    for sample in range(sample_num):
        for i in range(0, cols-1):  # 共输出 cols * 2 - 1 行
            diff = cols - i - 1  # 每一行的差
            for j in range(cols):  # 数组中每一个值的下标范围是0到cols
                k = j - diff  # 通过一个下标值计算另一个下标值
                if isValidIndex(k, rows):  # 剩下就是判断这些下标值是否满足当前的情况， 这一步不怎么好理解
                    tem_arr[sample, i, k] = cfc[sample, k, j]
            temp = tem_arr[sample, i]
            tem_mean[i, sample] = np.sum(temp)

    return tem_mean


def two_box(data_unnormal, data_normal, pic_title, label_normal, label_unnormal, y_axis_label):
    labels = [label_unnormal, label_normal]
    # 三个箱型图的颜色 RGB （均为0~1的数据）
    colors = ['#ff0000', '#008000'] #, '#0000ff'(202 / 255., 96 / 255., 17 / 255.), (255 / 255., 217 / 255., 102 / 255.), (137 / 255., 128 / 255., 68 / 255.)
    # plt.figure(figsize=(13, 10), dpi=80)
    for i in range(0, 9):
        data_unnormal_normal = [data_unnormal[i, :][data_unnormal[i, :]!=0], data_normal[i, :][data_normal[i, :]!=0]]
        # 绘制箱型图
        # patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
        bplot = plt.boxplot(data_unnormal_normal, patch_artist=True, labels=labels, positions=(i-5.8, i-5.4), widths=0.2, showfliers=False)# , showmeans=True
        # 将三个箱分别上色
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    x_position = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_position_fmt = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    plt.xticks([i - 6.5 for i in x_position], x_position_fmt)

    plt.ylabel(y_axis_label)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    plt.legend(bplot['boxes'], labels, bbox_to_anchor=(1, 1))  # 绘制表示框，右下角绘制
    plt.title("{}".format(pic_title))
    plt.savefig(fname="./box_plot/pic_{}.svg".format(pic_title), format='svg', bbox_inches='tight', dpi=500)
    plt.close("all")
    # plt.show()


def spdlogmap(spdmatrix):
    s, u = np.linalg.eig(spdmatrix)
    s = np.diag(np.log(s))
    output = u @ s @ u.T
    return output


def spdexpmap(input):
    s, u = np.linalg.eig(input)
    s = np.diag(np.exp(s))
    output = u @ s @ u.T
    return output


def log(X):
    S, U = X.symeig(eigenvectors=True)
    mask = (S <= 0).any(dim=-1)
    if mask.any():
        S_min, _ = S.min(dim=-1)
        S = S + ((1e-5 + abs(S_min)) * mask).unsqueeze(-1)

    S = S.log().diag_embed()
    return U @ S @ U.transpose(-2, -1)


def exp(X):
    S, U = X.symeig(eigenvectors=True)
    S = S.exp().diag_embed()
    return U @ S @ U.transpose(-2, -1)


def logm(x, y):
    c = x.cholesky()
    c_inv = c.inverse()
    return c @ log(c_inv @ y @ c_inv.transpose(-2, -1)) @ c.transpose(-2, -1)


def expm(x, y):
    c = x.cholesky()
    c_inv = c.inverse()
    return c @ exp(c_inv @ y @ c_inv.transpose(-2, -1)) @ c.transpose(-2, -1)


def riemannian_mean(spds, num_iter=100, eps_thresh=1e-4):
    mean = spds.mean(dim=-3).unsqueeze(-3)
    for iter in range(num_iter):
        tangent_mean = logm(mean, spds).mean(dim=-3).unsqueeze(-3)
        mean = expm(mean, tangent_mean)
        eps = tangent_mean.norm(p='fro', dim=(-2, -1)).mean()
        if eps < eps_thresh:
            break
    return mean.squeeze(-3)


def matrix_cfc_plot_weight_train(cfc_path, data_path, node_num, wavelets_num, weight):
    # count_num_unnormal = 0
    # count_num_normal = 0
    # for filename in os.listdir(cfc_path):
    #     abs_path = os.path.abspath(cfc_path)
    #     complete_path = os.path.join(abs_path, filename)
    #     cfc_label = open(complete_path, 'rb')
    #     cfc = pickle.load(cfc_label)
    #     for i in range(len(cfc['cfc'])):
    #         for all_node_num in range(node_num):
    #             if cfc['label'] == 0:
    #                 count_num_normal += 1
    #             elif cfc['label'] == 1:
    #                 count_num_unnormal += 1

    CN_num = 0
    AD_num = 0
    cfc_all_normal = [] # np.zeros((count_num_normal, wavelets_num, wavelets_num))
    cfc_all_unnormal = [] # np.zeros((count_num_unnormal, wavelets_num, wavelets_num))
    for filename, filename_o in zip(os.listdir(cfc_path), os.listdir(data_path)):
        abs_path = os.path.abspath(cfc_path)
        complete_path = os.path.join(abs_path, filename)
        cfc_label = open(complete_path, 'rb')
        cfc = pickle.load(cfc_label)
        for i in range(len(cfc['cfc'])):
            a = cfc['cfc'][i].tolist()
            cfc_matrix: np.array = np.array(a).astype('double')
            for all_node_num in range(node_num):
                if weight[all_node_num]>0:
                    if cfc['label'] == 0: # and CN_num < count_num_normal
                        cfc_all_normal.append(30*weight[all_node_num]*cfc_matrix[all_node_num]/weight.sum()) # cfc_all_normal[CN_num] = 90*weight[all_node_num]*cfc_matrix[all_node_num]/weight.sum()
                        CN_num += 1
                    elif cfc['label'] == 1: # and AD_num < count_num_unnormal
                        cfc_all_unnormal.append(30*weight[all_node_num]*cfc_matrix[all_node_num]/weight.sum()) # cfc_all_unnormal[AD_num] = 90*weight[all_node_num]*cfc_matrix[all_node_num]/weight.sum()
                        AD_num += 1
                    else:
                        print("AD or CN data is enough! Sample: ", filename)

    print(AD_num)
    print(CN_num)

    return cfc_all_normal, cfc_all_unnormal, CN_num, AD_num


# need to change the sliding mean
def matrix_cfc_plot(cfc_path, data_path, node_num, wavelets_num):
    count_num_unnormal = 0
    count_num_normal = 0
    for filename in os.listdir(cfc_path):
        abs_path = os.path.abspath(cfc_path)
        complete_path = os.path.join(abs_path, filename)
        cfc_label = open(complete_path, 'rb')
        cfc = pickle.load(cfc_label)
        for i in range(len(cfc['cfc'])):
            for all_node_num in range(node_num):
                if cfc['label'] == 0:
                    count_num_normal += 1
                elif cfc['label'] == 1:
                    count_num_unnormal += 1

    # cfc_ave_normal = np.zeros((wavelets_num, wavelets_num))
    # cfc_ave_unnormal = np.zeros((wavelets_num, wavelets_num))
    CN_num = 0
    AD_num = 0
    cfc_all_normal = np.zeros((count_num_normal, wavelets_num, wavelets_num))
    cfc_all_unnormal = np.zeros((count_num_unnormal, wavelets_num, wavelets_num))
    for filename, filename_o in zip(os.listdir(cfc_path), os.listdir(data_path)):
        abs_path = os.path.abspath(cfc_path)
        complete_path = os.path.join(abs_path, filename)
        cfc_label = open(complete_path, 'rb')
        cfc = pickle.load(cfc_label)
        # cfc_mat = joblib.load(complete_path)
        for i in range(len(cfc['cfc'])):
            print(len(cfc['cfc']))
            a = cfc['cfc'][i].tolist()
            cfc_matrix: np.array = np.array(a).astype('double')
            # cfc_mat_log = cfc_mat_log + np.log(cfc_matrix)
            # cfc_mat_a = cfc_mat_a + cfc_matrix
            # cfc_mat_b = cfc_mat_b + np.linalg.inv(cfc_matrix)
            # cfc_mat = spdlogmap(cfc_mat) + spdlogmap(cfc_matrix)
            # cfc_mat = spdexpmap(cfc_mat)
            for all_node_num in range(node_num):
                if cfc['label'] == 0 and CN_num < count_num_normal:
                    temp = cfc_matrix[all_node_num]
                    temp[temp < 0] = 0
                    cfc_all_normal[CN_num] = temp
                    # cfc_ave_normal_a = cfc_ave_normal_a + cfc_mat_a[all_node_num]
                    # cfc_ave_normal_b = cfc_ave_normal_b + cfc_mat_b[all_node_num]
                    # cfc_ave_normal_log = cfc_ave_normal_log + cfc_mat_log[all_node_num]
                    CN_num += 1
                elif cfc['label'] == 1 and AD_num < count_num_unnormal:
                    temp = cfc_matrix[all_node_num]
                    temp[temp < 0] = 0
                    cfc_all_unnormal[AD_num] = temp
                    # cfc_ave_unnormal_a = cfc_ave_unnormal_a + cfc_mat_a[all_node_num]
                    # cfc_ave_unnormal_b = cfc_ave_unnormal_b + cfc_mat_b[all_node_num]
                    # cfc_ave_unnormal_log = cfc_ave_unnormal_log + cfc_mat_log[all_node_num]
                    AD_num += 1
                else:
                    print("AD or CN data is enough! Sample: ", filename)

    print(AD_num)
    print(CN_num)

        # cfc_mat = np.real((cfc_mat_b**(-0.5))*(cfc_mat_b**0.5 * cfc_mat_a * cfc_mat_b**0.5)**0.5 * (cfc_mat_b**(-0.5)))
        # cfc_mat = cfc_mat / len(cfc['cfc'])
        # cfc = open(complete_path, 'rb')
        # cfc_mat = pickle.load(cfc)
        # abs_path_label = os.path.abspath(data_path)
        # complete_path_label = os.path.join(abs_path_label, filename_o)
        # cfc_label = open(complete_path_label, 'rb')
        # BOLD_window, label = pickle.load(cfc_label)

    # print(cfc_ave_normal_a, cfc_ave_normal_b, cfc_ave_unnormal_a, cfc_ave_unnormal_b, cfc_ave_normal_log, cfc_ave_unnormal_log)
    # for j in range(10):
    #     cfc_ave_normal_a[j, j] = 0
    #     cfc_ave_normal_b[j, j] = 0
    #     cfc_ave_unnormal_a[j, j] = 0
    #     cfc_ave_unnormal_b[j, j] = 0
    #     cfc_ave_normal_log[j, j] = 0
    #     cfc_ave_unnormal_log[j, j] = 0
    # v, q = np.linalg.eig(cfc_ave_normal_b)
    # V = np.diag(v ** (-0.5))
    # T = q * V * np.linalg.inv(q)
    # vu, qu = np.linalg.eig(cfc_ave_unnormal_b)
    # VU = np.diag(vu ** (-0.5))
    # TU = qu * VU * np.linalg.inv(qu)
    # cfc_ave_normal = np.real(T * fractional_matrix_power(fractional_matrix_power(cfc_ave_normal_b, 0.5) * cfc_ave_normal_a * fractional_matrix_power(cfc_ave_normal_b, 0.5), 0.5) * T)
    # cfc_ave_unnormal = np.real(TU * fractional_matrix_power(fractional_matrix_power(cfc_ave_unnormal_b, 0.5) * cfc_ave_unnormal_a * fractional_matrix_power(cfc_ave_unnormal_b, 0.5), 0.5) * TU)
    # print(cfc_ave_normal, cfc_ave_unnormal)
    # cfc_ave_normal = np.exp(cfc_ave_normal_log)
    # cfc_ave_unnormal = np.exp(cfc_ave_unnormal_log)
    # print(cfc_ave_normal, cfc_ave_unnormal)

    return cfc_all_normal, cfc_all_unnormal, CN_num, AD_num


def matrix_cfc_plot_task(cfc_path, data_path, node_num, wavelets_num):
    count_num_0bk = 0
    count_num_2bk = 0
    count_num_body = 0
    count_num_face = 0
    count_num_tools = 0
    count_num_place = 0
    for filename in os.listdir(cfc_path):
        abs_path = os.path.abspath(cfc_path)
        complete_path = os.path.join(abs_path, filename)
        cfc_label = open(complete_path, 'rb')
        cfc = pickle.load(cfc_label)
        for i in range(len(cfc['cfc'])):
            for all_node_num in range(node_num):
                if cfc['label'] == 0:
                    count_num_2bk += 1
                    count_num_body += 1
                elif cfc['label'] == 1:
                    count_num_0bk += 1
                    count_num_face += 1
                elif cfc['label'] == 2:
                    count_num_2bk += 1
                    count_num_tools += 1
                elif cfc['label'] == 3:
                    count_num_0bk += 1
                    count_num_body += 1
                elif cfc['label'] == 4:
                    count_num_0bk += 1
                    count_num_place += 1
                elif cfc['label'] == 5:
                    count_num_2bk += 1
                    count_num_face += 1
                elif cfc['label'] == 6:
                    count_num_0bk += 1
                    count_num_tools += 1
                elif cfc['label'] == 7:
                    count_num_2bk += 1
                    count_num_place += 1

    num_0bk = 0
    num_2bk = 0
    num_body = 0
    num_face = 0
    num_tools = 0
    num_place = 0
    cfc_all_0bk = np.zeros((count_num_0bk, wavelets_num, wavelets_num))
    cfc_all_2bk = np.zeros((count_num_2bk, wavelets_num, wavelets_num))
    cfc_all_body = np.zeros((count_num_body, wavelets_num, wavelets_num))
    cfc_all_face = np.zeros((count_num_face, wavelets_num, wavelets_num))
    cfc_all_tools = np.zeros((count_num_tools, wavelets_num, wavelets_num))
    cfc_all_place = np.zeros((count_num_place, wavelets_num, wavelets_num))
    for filename, filename_o in zip(os.listdir(cfc_path), os.listdir(data_path)):
        abs_path = os.path.abspath(cfc_path)
        complete_path = os.path.join(abs_path, filename)
        cfc_label = open(complete_path, 'rb')
        cfc = pickle.load(cfc_label)
        for i in range(len(cfc['cfc'])):
            print(len(cfc['cfc']))
            a = cfc['cfc'][i].tolist()
            cfc_matrix: np.array = np.array(a).astype('double')
            for all_node_num in range(node_num):
                if cfc['label'] == 0:
                    cfc_all_2bk[num_2bk] = cfc_matrix[all_node_num]
                    cfc_all_body[num_body] = cfc_matrix[all_node_num]
                    num_2bk += 1
                    num_body += 1
                elif cfc['label'] == 1:
                    cfc_all_0bk[num_0bk] = cfc_matrix[all_node_num]
                    cfc_all_face[num_face] = cfc_matrix[all_node_num]
                    num_0bk += 1
                    num_face += 1
                elif cfc['label'] == 2:
                    cfc_all_2bk[num_2bk] = cfc_matrix[all_node_num]
                    cfc_all_tools[num_tools] = cfc_matrix[all_node_num]
                    num_2bk += 1
                    num_tools += 1
                elif cfc['label'] == 3:
                    cfc_all_0bk[num_0bk] = cfc_matrix[all_node_num]
                    cfc_all_body[num_body] = cfc_matrix[all_node_num]
                    num_0bk += 1
                    num_body += 1
                elif cfc['label'] == 4:
                    cfc_all_0bk[num_0bk] = cfc_matrix[all_node_num]
                    cfc_all_place[num_place] = cfc_matrix[all_node_num]
                    num_0bk += 1
                    num_place += 1
                elif cfc['label'] == 5:
                    cfc_all_2bk[num_2bk] = cfc_matrix[all_node_num]
                    cfc_all_face[num_face] = cfc_matrix[all_node_num]
                    num_2bk += 1
                    num_face += 1
                elif cfc['label'] == 6:
                    cfc_all_0bk[num_0bk] = cfc_matrix[all_node_num]
                    cfc_all_tools[num_tools] = cfc_matrix[all_node_num]
                    num_0bk += 1
                    num_tools += 1
                elif cfc['label'] == 7:
                    cfc_all_2bk[num_2bk] = cfc_matrix[all_node_num]
                    cfc_all_place[num_place] = cfc_matrix[all_node_num]
                    num_2bk += 1
                    num_place += 1
                else:
                    print("AD or CN data is enough! Sample: ", filename)

    print(num_0bk)
    print(num_2bk)
    print(num_body)
    print(num_face)
    print(num_place)
    print(num_tools)

    return cfc_all_0bk, cfc_all_2bk, cfc_all_body, cfc_all_face, cfc_all_tools, cfc_all_place, num_0bk, num_2bk, num_body, num_face, num_tools, num_place


def matrix_cfc_plot_task_single_label(cfc_path, data_path, node_num, wavelets_num):
    # count_num = np.zeros(8)
    count_num_a = 0
    count_num_b = 0
    count_num_c = 0
    count_num_d = 0
    count_num_e = 0
    count_num_f = 0
    count_num_g = 0
    count_num_h = 0
    for filename in os.listdir(cfc_path):
        abs_path = os.path.abspath(cfc_path)
        complete_path = os.path.join(abs_path, filename)
        cfc_label = open(complete_path, 'rb')
        cfc = pickle.load(cfc_label)
        for i in range(len(cfc['cfc'])):
            for all_node_num in range(node_num):
                if cfc['label'] == 0:
                    count_num_a += 1
                elif cfc['label'] == 1:
                    count_num_b += 1
                elif cfc['label'] == 2:
                    count_num_c += 1
                elif cfc['label'] == 3:
                    count_num_d += 1
                elif cfc['label'] == 4:
                    count_num_e += 1
                elif cfc['label'] == 5:
                    count_num_f += 1
                elif cfc['label'] == 6:
                    count_num_g += 1
                elif cfc['label'] == 7:
                    count_num_h += 1

    # cfc_ave_normal = np.zeros((wavelets_num, wavelets_num))
    # cfc_ave_unnormal = np.zeros((wavelets_num, wavelets_num))
    a_num = 0
    b_num = 0
    c_num = 0
    d_num = 0
    e_num = 0
    f_num = 0
    g_num = 0
    h_num = 0
    cfc_all_a = np.zeros((count_num_a, wavelets_num, wavelets_num))
    cfc_all_b = np.zeros((count_num_b, wavelets_num, wavelets_num))
    cfc_all_c = np.zeros((count_num_c, wavelets_num, wavelets_num))
    cfc_all_d = np.zeros((count_num_d, wavelets_num, wavelets_num))
    cfc_all_e = np.zeros((count_num_e, wavelets_num, wavelets_num))
    cfc_all_f = np.zeros((count_num_f, wavelets_num, wavelets_num))
    cfc_all_g = np.zeros((count_num_g, wavelets_num, wavelets_num))
    cfc_all_h = np.zeros((count_num_h, wavelets_num, wavelets_num))
    for filename, filename_o in zip(os.listdir(cfc_path), os.listdir(data_path)):
        abs_path = os.path.abspath(cfc_path)
        complete_path = os.path.join(abs_path, filename)
        cfc_label = open(complete_path, 'rb')
        cfc = pickle.load(cfc_label)
        # cfc_mat = joblib.load(complete_path)
        for i in range(len(cfc['cfc'])):
            print(len(cfc['cfc']))
            a = cfc['cfc'][i].tolist()
            cfc_matrix: np.array = np.array(a).astype('double')
            for all_node_num in range(node_num):
                if cfc['label'] == 0:
                    cfc_all_a[a_num] = cfc_matrix[all_node_num]
                    a_num += 1
                elif cfc['label'] == 1:
                    cfc_all_b[b_num] = cfc_matrix[all_node_num]
                    b_num += 1
                elif cfc['label'] == 2:
                    cfc_all_c[c_num] = cfc_matrix[all_node_num]
                    c_num += 1
                elif cfc['label'] == 3:
                    cfc_all_d[d_num] = cfc_matrix[all_node_num]
                    d_num += 1
                elif cfc['label'] == 4:
                    cfc_all_e[e_num] = cfc_matrix[all_node_num]
                    e_num += 1
                elif cfc['label'] == 5:
                    cfc_all_f[f_num] = cfc_matrix[all_node_num]
                    f_num += 1
                elif cfc['label'] == 6:
                    cfc_all_g[g_num] = cfc_matrix[all_node_num]
                    g_num += 1
                elif cfc['label'] == 7:
                    cfc_all_h[h_num] = cfc_matrix[all_node_num]
                    h_num += 1
                else:
                    print("AD or CN data is enough! Sample: ", filename)
                    print(cfc['label'])

    print(a_num)
    print(b_num)
    print(c_num)
    print(d_num)
    print(e_num)
    print(f_num)
    print(g_num)
    print(h_num)

    cfc_ave_a = np.mean(cfc_all_a, axis=0)
    cfc_ave_b = np.mean(cfc_all_b, axis=0)
    cfc_ave_c = np.mean(cfc_all_c, axis=0)
    cfc_ave_d = np.mean(cfc_all_d, axis=0)
    cfc_ave_e = np.mean(cfc_all_e, axis=0)
    cfc_ave_f = np.mean(cfc_all_f, axis=0)
    cfc_ave_g = np.mean(cfc_all_g, axis=0)
    cfc_ave_h = np.mean(cfc_all_h, axis=0)
    np.savetxt('./data/cfc_ave_a_task.txt', cfc_ave_a, fmt='%f', delimiter=' ')
    np.savetxt('./data/cfc_ave_b_task.txt', cfc_ave_b, fmt='%f', delimiter=' ')
    np.savetxt('./data/cfc_ave_c_task.txt', cfc_ave_c, fmt='%f', delimiter=' ')
    np.savetxt('./data/cfc_ave_d_task.txt', cfc_ave_d, fmt='%f', delimiter=' ')
    np.savetxt('./data/cfc_ave_e_task.txt', cfc_ave_e, fmt='%f', delimiter=' ')
    np.savetxt('./data/cfc_ave_f_task.txt', cfc_ave_f, fmt='%f', delimiter=' ')
    np.savetxt('./data/cfc_ave_g_task.txt', cfc_ave_g, fmt='%f', delimiter=' ')
    np.savetxt('./data/cfc_ave_h_task.txt', cfc_ave_h, fmt='%f', delimiter=' ')
    plot_mat_pattern(cfc_ave_a, cfc_ave_b, "a", "b", "task1")
    plot_mat_pattern(cfc_ave_c, cfc_ave_d, "c", "d", "task2")
    plot_mat_pattern(cfc_ave_e, cfc_ave_f, "e", "f", "task3")
    plot_mat_pattern(cfc_ave_g, cfc_ave_h, "g", "h", "task4")


def icc_calculate(Y, icc_type):
    [n, k] = Y.shape

    # 自由度
    dfall = n * k - 1  # 所有自由度
    dfe = (n - 1) * (k - 1)  # 剩余自由度
    dfc = k - 1  # 列自由度
    dfr = n - 1  # 行自由度

    # 所有的误差
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # 误差均方
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
    )
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()
    MSE = SSE / dfe

    # 列均方
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc

    # 行均方
    SSR = ((np.mean(Y, 1) - mean_Y) ** 2).sum() * k
    MSR = SSR / dfr

    if icc_type == "icc(1)":
        SSW = SST - SSR  # 剩余均方
        MSW = SSW / (dfall - dfr)
        ICC1 = (MSR - MSW) / (MSR + (k - 1) * MSW)
        ICC2 = (MSR - MSW) / MSR
    elif icc_type == "icc(2)":
        ICC1 = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
        ICC2 = (MSR - MSE) / (MSR + (MSC - MSE) / n)
    elif icc_type == "icc(3)":
        ICC1 = (MSR - MSE) / (MSR + (k - 1) * MSE)
        ICC2 = (MSR - MSE) / MSR

    return ICC1, ICC2


def plot_mat_pattern(cfc_ave_normal, cfc_ave_unnormal, cfc_normal_title, cfc_unnormal_title, pic_title):
    print(cfc_ave_unnormal)
    for i in range(10):
        cfc_ave_unnormal[i, i] = 0
        cfc_ave_normal[i, i] = 0

    print(cfc_ave_normal, cfc_ave_unnormal)
    mins = []
    maxes = []
    mins.append(np.min(cfc_ave_unnormal))
    mins.append(np.min(cfc_ave_normal))
    maxes.append(np.max(cfc_ave_unnormal))
    maxes.append(np.max(cfc_ave_normal))
    vmin = np.min(mins)
    vmax = np.max(maxes)
    # vmin = np.min(np.min(cfc_ave_unnormal), np.min(cfc_ave_normal))
    # vmax = np.max(np.max(cfc_ave_unnormal), np.max(cfc_ave_normal))
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(4, 2))
    # fig.subplots_adjust(hspace=0.1)
    xlabels = '1 2 3 4 5 6 7 8 9 10'.split()
    ylabels = '1 2 3 4 5 6 7 8 9 10'.split()
    ax = axs[0]
    im = ax.imshow(cfc_ave_unnormal, vmin=0, vmax=0.2, cmap='RdBu')#, cmap='Blues' 0.06
    ax.set_title(cfc_unnormal_title, fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    ax = axs[1]
    im = ax.imshow(cfc_ave_normal, vmin=0, vmax=0.2, cmap='RdBu')#, cmap='Blues'
    ax.set_title(cfc_normal_title, fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    plt.xticks(range(len(xlabels)), xlabels, fontproperties='Times New Roman', size=12)
    plt.yticks(range(len(ylabels)), ylabels, fontproperties='Times New Roman', size=12)
    cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax, ticks=[0, 0.1, 0.2])
    cbar_ax = cbar.ax
    cbar_ax.tick_params(labelsize=12)
    # cbar_ax.tick_params(which='major', direction='in', labelsize=12, length=7.5)
    fig.savefig('./box_plot/cfc_{}.svg'.format(pic_title),
                    format='svg', bbox_inches='tight', dpi=500) # D:/research/Nonliner Dimensional Reduction/4_GNN_SPD/CFC matrix/box_plot/cfc_{}.svg
    plt.close('all')


def plot_mat_pattern_task(cfc_ave_a, cfc_ave_b, cfc_ave_c, cfc_ave_d, cfc_ave_e, cfc_ave_f, cfc_ave_g, cfc_ave_h, cfc_a_title, cfc_b_title, cfc_c_title, cfc_d_title, cfc_e_title, cfc_f_title, cfc_g_title, cfc_h_title, pic_title):
    for i in range(10):
        cfc_ave_a[i, i] = 0
        cfc_ave_b[i, i] = 0
        cfc_ave_c[i, i] = 0
        cfc_ave_d[i, i] = 0
        cfc_ave_e[i, i] = 0
        cfc_ave_f[i, i] = 0
        cfc_ave_g[i, i] = 0
        cfc_ave_h[i, i] = 0

    mins = []
    maxes = []
    mins.append(np.min(cfc_ave_a))
    mins.append(np.min(cfc_ave_b))
    maxes.append(np.max(cfc_ave_a))
    maxes.append(np.max(cfc_ave_b))
    mins.append(np.min(cfc_ave_c))
    mins.append(np.min(cfc_ave_d))
    maxes.append(np.max(cfc_ave_c))
    maxes.append(np.max(cfc_ave_d))
    mins.append(np.min(cfc_ave_e))
    mins.append(np.min(cfc_ave_f))
    maxes.append(np.max(cfc_ave_e))
    maxes.append(np.max(cfc_ave_f))
    mins.append(np.min(cfc_ave_g))
    mins.append(np.min(cfc_ave_h))
    maxes.append(np.max(cfc_ave_g))
    maxes.append(np.max(cfc_ave_h))
    vmin = np.min(mins)
    vmax = np.max(maxes)
    # vmin = np.min(np.min(cfc_ave_unnormal), np.min(cfc_ave_normal))
    # vmax = np.max(np.max(cfc_ave_unnormal), np.max(cfc_ave_normal))
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(4, 8))
    # fig.subplots_adjust(hspace=0.1)
    xlabels = '1 2 3 4 5 6 7 8 9 10'.split()
    ylabels = '1 2 3 4 5 6 7 8 9 10'.split()
    ax = axs[0]
    im = ax.imshow(cfc_ave_a, vmin=vmin, vmax=vmax, cmap='RdBu')#, cmap='Blues'
    ax.set_title(cfc_a_title, fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    ax = axs[1]
    im = ax.imshow(cfc_ave_b, vmin=vmin, vmax=vmax, cmap='RdBu')#, cmap='Blues'
    ax.set_title(cfc_b_title, fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    ax = axs[2]
    im = ax.imshow(cfc_ave_c, vmin=vmin, vmax=vmax, cmap='RdBu')  # , cmap='Blues'
    ax.set_title(cfc_c_title, fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    ax = axs[3]
    im = ax.imshow(cfc_ave_d, vmin=vmin, vmax=vmax, cmap='RdBu')  # , cmap='Blues'
    ax.set_title(cfc_d_title, fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    ax = axs[4]
    im = ax.imshow(cfc_ave_e, vmin=vmin, vmax=vmax, cmap='RdBu')  # , cmap='Blues'
    ax.set_title(cfc_e_title, fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    ax = axs[5]
    im = ax.imshow(cfc_ave_f, vmin=vmin, vmax=vmax, cmap='RdBu')  # , cmap='Blues'
    ax.set_title(cfc_f_title, fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    ax = axs[6]
    im = ax.imshow(cfc_ave_g, vmin=vmin, vmax=vmax, cmap='RdBu')  # , cmap='Blues'
    ax.set_title(cfc_g_title, fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    ax = axs[7]
    im = ax.imshow(cfc_ave_h, vmin=vmin, vmax=vmax, cmap='RdBu')  # , cmap='Blues'
    ax.set_title(cfc_h_title, fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    plt.xticks(range(len(xlabels)), xlabels, fontproperties='Times New Roman', size=12)
    plt.yticks(range(len(ylabels)), ylabels, fontproperties='Times New Roman', size=12)
    cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar_ax = cbar.ax
    cbar_ax.tick_params(labelsize=12)
    # cbar_ax.tick_params(which='major', direction='in', labelsize=12, length=7.5)
    fig.savefig('./box_plot/cfc_{}.svg'.format(pic_title),
                    format='svg', bbox_inches='tight', dpi=500) # D:/research/Nonliner Dimensional Reduction/4_GNN_SPD/CFC matrix/box_plot/cfc_{}.svg
    plt.close('all')


def plot_training_pattern(training_data_path, testing_data_path, o_data_path, to_data_path):
    train_results = []
    current_address = os.path.dirname(os.path.abspath(training_data_path))
    for parent, dirnames, filenames in os.walk(current_address):
        for dirname in dirnames:
            str = re.compile(o_data_path)
            match_obj = re.findall(str, dirname)
            if match_obj:
                train_results.append(joblib.load(os.path.join(parent, dirname, f'result.pkl')))
                print(os.path.join(parent, dirname, f'result.pkl'))
    test_results = []
    current_address = os.path.dirname(os.path.abspath(testing_data_path))
    for parent, dirnames, filenames in os.walk(current_address):
        for dirname in dirnames:
            str = re.compile(to_data_path)
            match_obj = re.findall(str, dirname)
            if match_obj:
                test_results.append(joblib.load(os.path.join(parent, dirname, f'result.pkl')))
                print(os.path.join(parent, dirname, f'result.pkl'))
    # train_results = joblib.load(training_data_path)
    # test_results = joblib.load(testing_data_path)
    cfc_ave_ad = np.zeros((10, 10))
    cfc_ave_cn = np.zeros((10, 10))
    cfc_all_ad = np.zeros((69, 10, 10)) # aal15 294 hcp 69
    cfc_all_cn = np.zeros((75, 10, 10)) # aal15 546 hcp 75
    test_num_ad = 0
    test_num_cn = 0
    # for i in range(len(train_results) - len(test_results)):
    #     if i % 45 > 29:
    #         a = train_results[i]['gam_output'].tolist()
    #         train_cfc: np.array = np.array(a).astype('double')
            # cfc_ave_ad = cfc_ave_ad + train_cfc[0]
            # test_num_ad = test_num_ad + 1
            # cfc_ave_cn = cfc_ave_cn + train_cfc[0]
            # test_num_cn = test_num_cn + 1
            # if int(test_results[i-len(train_results)+len(test_results)]['target']) == 1:
            #     cfc_ave_ad = cfc_ave_ad + train_cfc[0]
            #     test_num_ad = test_num_ad + 1
            # else:
            #     cfc_ave_cn = cfc_ave_cn + train_cfc[0]
            #     test_num_cn = test_num_cn + 1

    # for i in range(len(train_results)-len(test_results), len(train_results)):
    #     if (i-len(train_results)+len(test_results))%60>39:
    for i in range(len(train_results)):
        a = train_results[i]['gam_output'].tolist()
        train_cfc: np.array = np.array(a).astype('double')
        # if int(test_results[i-len(train_results)+len(test_results)]['target']) == 1:
        if int(train_results[i]['target']) == 1:
            cfc_ave_ad = cfc_ave_ad + train_cfc[0]
            cfc_all_ad[test_num_ad] = train_cfc[0]
            test_num_ad = test_num_ad + 1
        else:
            cfc_ave_cn = cfc_ave_cn + train_cfc[0]
            cfc_all_cn[test_num_cn] = train_cfc[0]
            test_num_cn = test_num_cn + 1

    cfc_ave_ad = cfc_ave_ad / test_num_ad
    cfc_ave_cn = cfc_ave_cn / test_num_cn
    print(test_num_ad)
    print(test_num_cn)

    for i in range(10):
        cfc_ave_ad[i, i] = 0
        cfc_ave_cn[i, i] = 0

    mins = []
    maxes = []
    mins.append(np.min(cfc_ave_ad))
    mins.append(np.min(cfc_ave_cn))
    maxes.append(np.max(cfc_ave_ad))
    maxes.append(np.max(cfc_ave_cn))
    vmin = np.min(mins)
    vmax = np.max(maxes)
    # vmin = np.min(np.min(cfc_ave_ad), np.min(cfc_ave_cn))
    # vmax = np.max(np.max(cfc_ave_ad), np.max(cfc_ave_cn))
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(4, 2))
    # fig.subplots_adjust(hspace=0.1)
    xlabels = '1 2 3 4 5 6 7 8 9 10'.split()
    ylabels = '1 2 3 4 5 6 7 8 9 10'.split()
    ax = axs[0]
    im = ax.imshow(cfc_ave_ad, vmin=vmin, vmax=vmax, cmap='RdBu')  # , cmap='Blues'
    ax.set_title('AD', fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    ax = axs[1]
    im = ax.imshow(cfc_ave_cn, vmin=vmin, vmax=vmax, cmap='RdBu')  # , cmap='Blues'
    ax.set_title('CN', fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels]
    plt.xticks(range(len(xlabels)), xlabels, fontproperties='Times New Roman', size=12)
    plt.yticks(range(len(ylabels)), ylabels, fontproperties='Times New Roman', size=12)
    cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar_ax = cbar.ax
    cbar_ax.tick_params(labelsize=12)
    # cbar_ax.tick_params(which='major', direction='in', labelsize=12, length=7.5)
    fig.savefig('./box_plot/training_cfc_pattern_{}.svg'.format(o_data_path), format='svg', bbox_inches='tight', dpi=500)
    plt.close('all')

    return cfc_all_ad, cfc_all_cn


def plot_training_pattern_newformat(training_data_path, testing_data_path, o_data_path, to_data_path):
    test_results = []
    acc_results = []
    subject_set = []
    cfc_all_ad = []
    cfc_all_cn = []
    step = []
    walk = []
    prediction = []
    true = []
    # abs_path_test = os.path.abspath(o_data_path)
    # for filenames_test in os.listdir(abs_path_test):
    abs_path = os.path.abspath(to_data_path) # 文件夹的绝对路径
    print(abs_path)
    for filenames in os.listdir(abs_path):
        # print(filenames)
        # filenames = 'result_9_93.pkl'
        split_filename = filenames.split('_')
        subject = split_filename[1]
        subject_set.append(subject)
        repeat_time = split_filename[-1]
        repeat_time = re.findall(r"\d*", repeat_time)
        repeat_time = int(repeat_time[0])
        path_parameter = os.path.split(abs_path)
        parameter = path_parameter[1].split('_')
        repeats = parameter[0]
        repeatst = parameter[1]
        times = parameter[2]
        if repeat_time > int(repeats):
            complete_filename = os.path.join(abs_path, filenames)  # 将路径与文件名结合起来就是每个文件的完整路径
            acc_cfc = open(complete_filename, 'rb')
            acc_results.append(joblib.load(acc_cfc))
        if repeat_time > 0 and repeat_time < int(repeats): # int(repeats)-15: # :
            complete_filename = os.path.join(abs_path, filenames)  # 将路径与文件名结合起来就是每个文件的完整路径
            test_cfc = open(complete_filename, 'rb')
            test_results.append(joblib.load(test_cfc))
            # a, b, c, d, e, f, g, h, i = test_results[0]
    for i in range(len(acc_results)):
        prediction.append(int(torch.argmax(acc_results[i]['logits'])))
        true.append(int(acc_results[i]['target']))
    print(confusion_matrix(np.array(true), np.array(prediction)))
    conf = confusion_matrix(np.array(true), np.array(prediction))
    print("acc: {}".format((conf[0, 0]+conf[1, 1])/(sum(sum(conf)))))
    print("sens: {}".format(conf[1, 1]/(conf[0, 1]+conf[1, 1])))
    print("spec: {}".format(conf[0, 0]/(conf[0, 0]+conf[0, 1])))
    for i in range(len(test_results)):
        node_count = Counter(test_results[i]['train_steps'])
        if max(node_count.values())<60:
            for j in range(int(times)):
                if j>=int(times)-60:
                    a = test_results[i]['gam_output'][j].tolist()
                    train_cfc: np.array = np.array(a).astype('double')
                    b = test_results[i]['train_steps'][j]
                    # train_step: np.array = np.array(b).astype('double')
                    # if int(test_results[i-len(train_results)+len(test_results)]['target']) == 1:
                    c = [test_results[i]['train_steps'][j-1],test_results[i]['train_steps'][j]]
                    step.append(int(b))
                    walk.append(str(c))
                    if int(test_results[i]['target']) == 1:
                        # cfc_all_ad[test_num_ad] = train_cfc[0]
                        cfc_all_ad.append(train_cfc[0])
                        # test_num_ad = test_num_ad + 1
                    else:
                        # cfc_all_cn[test_num_cn] = train_cfc[0]
                        cfc_all_cn.append(train_cfc[0])
                        # test_num_cn = test_num_cn + 1

    return cfc_all_ad, cfc_all_cn, step, walk # , test_num_ad, test_num_cn

if __name__ == '__main__':
    args = parameter_parser()

    # # draw the spd distance of cfc
    # cfc_dis_pathcn = "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5_dis\\sub-002S0685_aal.txt_4.pkl"
    # cfc_dis_pathsmc = "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5_dis\\sub-002S5178_aal.txt_21.pkl"
    # cfc_dis_pathlmci = "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5_dis\\sub-002S1155_aal.txt_6.pkl"
    # cfc_dis_pathad = "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5_dis\\sub-006S4192_aal.txt_26.pkl"
    # cfc_dis_path = [cfc_dis_pathcn, cfc_dis_pathsmc, cfc_dis_pathlmci, cfc_dis_pathad]
    # for filename in cfc_dis_path:
    #     cfc_label = open(filename, 'rb')
    #     cfc = pickle.load(cfc_label)
    #     cfc_all_normal = np.zeros((len(cfc), 90, 90))
    #     CN_num = 0
    #     for i in range(len(cfc)):
    #         a = cfc[i].tolist()
    #         cfc_matrix: np.array = np.array(a).astype('double')
    #         cfc_all_normal[CN_num] = cfc_matrix
    #         CN_num += 1
    #     cfc_ave_normal = np.mean(cfc_all_normal, axis=0)
    #     if filename == cfc_dis_pathcn:
    #         cfc_ave_cn = cfc_ave_normal
    #     elif filename == cfc_dis_pathlmci:
    #         cfc_ave_lmci = cfc_ave_normal
    #     elif filename == cfc_dis_pathsmc:
    #         cfc_ave_smc = cfc_ave_normal
    #     else:
    #         cfc_ave_ad = cfc_ave_normal
    # plot_mat_pattern(cfc_ave_cn, cfc_ave_smc, "CN", "SMC", "DIS1")
    # plot_mat_pattern(cfc_ave_lmci, cfc_ave_ad, "LMCI", "AD", "DIS2")

    # # calculate the pearson correlation coefficient of original bold data
    # cfc_dis_pathcn = "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\AAL_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5\\sub-002S0685_aal.txt_4.pkl"
    # cfc_dis_pathsmc = "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\AAL_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5\\sub-002S5178_aal.txt_21.pkl"
    # cfc_dis_pathlmci = "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\AAL_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5\\sub-002S1155_aal.txt_6.pkl"
    # cfc_dis_pathad = "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\AAL_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5\\sub-006S4192_aal.txt_26.pkl"
    # cfc_dis_path = [cfc_dis_pathcn, cfc_dis_pathsmc, cfc_dis_pathlmci, cfc_dis_pathad]
    # for filename in cfc_dis_path:
    #     bold_label = open(filename, 'rb')
    #     bold, label = pickle.load(bold_label)
    #     bold_matrix = np.float32(bold)
    #     if filename == cfc_dis_pathcn:
    #         cfc_ave_cn = corrcoef(bold_matrix)
    #     elif filename == cfc_dis_pathlmci:
    #         cfc_ave_lmci = corrcoef(bold_matrix)
    #     elif filename == cfc_dis_pathsmc:
    #         cfc_ave_smc = corrcoef(bold_matrix)
    #     else:
    #         cfc_ave_ad = corrcoef(bold_matrix)
    # plot_mat_pattern(cfc_ave_cn, cfc_ave_smc, "CN", "SMC", "graph1")
    # plot_mat_pattern(cfc_ave_lmci, cfc_ave_ad, "LMCI", "AD", "graph2")

    if args.task_pattern:
        cfc_all_0bk, cfc_all_2bk, cfc_all_body, cfc_all_face, cfc_all_tools, cfc_all_place, sample_num_0bk, sample_num_2bk, sample_num_body, sample_num_face, sample_num_tools, sample_num_place = matrix_cfc_plot_task(args.cfc_data_path, args.processed_data_path, args.node_num, args.wavelets_num)
        matrix_cfc_plot_task_single_label(args.cfc_data_path, args.processed_data_path, args.node_num, args.wavelets_num)
        cfc_ave_0bk = np.mean(cfc_all_0bk, axis=0)
        cfc_ave_2bk = np.mean(cfc_all_2bk, axis=0)
        cfc_ave_body = np.mean(cfc_all_body, axis=0)
        cfc_ave_face = np.mean(cfc_all_face, axis=0)
        cfc_ave_tools = np.mean(cfc_all_tools, axis=0)
        cfc_ave_place = np.mean(cfc_all_place, axis=0)
        np.savetxt('./data/cfc_ave_0bk.txt', cfc_ave_0bk, fmt='%f', delimiter=' ')
        np.savetxt('./data/cfc_ave_2bk.txt', cfc_ave_2bk, fmt='%f', delimiter=' ')
        np.savetxt('./data/cfc_ave_body.txt', cfc_ave_body, fmt='%f', delimiter=' ')
        np.savetxt('./data/cfc_ave_face.txt', cfc_ave_face, fmt='%f', delimiter=' ')
        np.savetxt('./data/cfc_ave_tools.txt', cfc_ave_tools, fmt='%f', delimiter=' ')
        np.savetxt('./data/cfc_ave_place.txt', cfc_ave_place, fmt='%f', delimiter=' ')
        # if args.task_boxplot:
        _, icc_0bk = get_diag_value(cfc_all_0bk)
        _, icc_2bk = get_diag_value(cfc_all_2bk)
        _, icc_body = get_diag_value(cfc_all_body)
        _, icc_face = get_diag_value(cfc_all_face)
        _, icc_tools = get_diag_value(cfc_all_tools)
        _, icc_place = get_diag_value(cfc_all_place)
        np.savetxt('./data/icc_0bk.txt', icc_0bk, fmt='%f', delimiter=' ')
        np.savetxt('./data/icc_2bk.txt', icc_2bk, fmt='%f', delimiter=' ')
        np.savetxt('./data/icc_body.txt', icc_body, fmt='%f', delimiter=' ')
        np.savetxt('./data/icc_face.txt', icc_face, fmt='%f', delimiter=' ')
        np.savetxt('./data/icc_tools.txt', icc_tools, fmt='%f', delimiter=' ')
        np.savetxt('./data/icc_place.txt', icc_place, fmt='%f', delimiter=' ')
        two_box(np.abs(icc_0bk), np.abs(icc_2bk), "task_abs_bk", "2bk", "0bk", "ICC Distribution")
        two_box(np.abs(icc_body), np.abs(icc_face), "task_abs_sub1", "face", "body", "ICC Distribution")
        two_box(np.abs(icc_tools), np.abs(icc_place), "task_abs_sub2", "place", "tools", "ICC Distribution")

    if not args.training_pattern and not args.task_pattern:
        cfc_all_normal, cfc_all_unnormal, sample_normal_num, sample_unnormal_num = matrix_cfc_plot(args.cfc_data_path, args.processed_data_path, args.node_num, args.wavelets_num)# 'D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\data\\hcp_cfc', AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5/aal_cn_ad_cfc_iter1000/hcp_cfc

        cfc_all_normal_stack = cfc_all_normal
        cfc_all_unnormal_stack = cfc_all_unnormal
        if args.riemannian:
            cfc_ave_normal = riemannian_mean(torch.Tensor(cfc_all_normal))
            cfc_ave_unnormal = riemannian_mean(torch.Tensor(cfc_all_unnormal))
            cfc_ave_normal = cfc_ave_normal.numpy()
            cfc_ave_unnormal = cfc_ave_unnormal.numpy()
        else:
            cfc_ave_normal = np.mean(cfc_all_normal, axis=0)
            cfc_ave_unnormal = np.mean(cfc_all_unnormal, axis=0)
        print(cfc_ave_normal, cfc_ave_unnormal)
        np.savetxt('./data/cfc_ave_unnormal.txt', cfc_ave_unnormal, fmt='%f', delimiter=' ')
        np.savetxt('./data/cfc_ave_normal.txt', cfc_ave_normal, fmt='%f', delimiter=' ')
        # cfc_all_normal_stack = np.zeros((args.node_num*args.count_num_normal, args.wavelets_num, args.wavelets_num))
        # cfc_all_unnormal_stack = np.zeros((args.node_num*args.count_num_unnormal, args.wavelets_num, args.wavelets_num))
        # for sample_i in range(args.count_num_unnormal):
        #     for node in range(args.node_num):
        #         cfc_all_unnormal_stack[node+args.node_num*sample_i] = cfc_all_unnormal[node, sample_i]
        #
        # for sample_i in range(args.count_num_normal):
        #     for node in range(args.node_num):
        #         cfc_all_normal_stack[node+args.node_num*sample_i] = cfc_all_normal[node, sample_i]

    if args.box_plot:
        # var_normal, icc_normal = get_diag_value(cfc_all_normal_stack)
        # var_unnormal, icc_unnormal = get_diag_value(cfc_all_unnormal_stack)
        mean_normal = get_diag_mean(cfc_all_normal_stack)
        mean_unnormal = get_diag_mean(cfc_all_unnormal_stack)
        # np.savetxt('./data/var_unnormal.txt', var_unnormal, fmt='%f', delimiter=' ')
        # np.savetxt('./data/var_normal.txt', var_normal, fmt='%f', delimiter=' ')
        # np.savetxt('./data/icc_unnormal.txt', icc_unnormal, fmt='%f', delimiter=' ')
        # np.savetxt('./data/icc_normal.txt', icc_normal, fmt='%f', delimiter=' ')
        np.savetxt('./data/mean_unnormal.txt', mean_unnormal, fmt='%f', delimiter=' ')
        np.savetxt('./data/mean_normal.txt', mean_normal, fmt='%f', delimiter=' ')
        if args.aal:
            # two_box(np.abs(var_unnormal), np.abs(var_normal), "aal_abs_var_15", "CN", "AD", "VAR Distribution")
            # two_box(np.abs(icc_unnormal), np.abs(icc_normal), "aal_abs_icc_15", "CN", "AD", "ICC Distribution")
            two_box(np.abs(mean_unnormal), np.abs(mean_normal), "aal_abs_mean_15", "CN", "AD", "Mean Distribution")
        elif args.hcp:
            two_box(np.abs(var_unnormal), np.abs(var_normal), "hcp_abs_var", "Unaffected", "Schezophrannia", "VAR Distribution")
            two_box(np.abs(icc_unnormal), np.abs(icc_normal), "hcp_abs_icc", "Unaffected", "Schezophrannia", "ICC Distribution")
            two_box(np.abs(mean_unnormal), np.abs(mean_normal), "hcp_abs_mean", "Unaffected", "Schezophrannia", "Mean Distribution")
        elif args.ocd:
            two_box(np.abs(var_unnormal), np.abs(var_normal), "ocd_abs_var", "CN", "OCD", "VAR Distribution")
            two_box(np.abs(icc_unnormal), np.abs(icc_normal), "ocd_abs_icc", "CN", "OCD", "ICC Distribution")
            two_box(np.abs(mean_unnormal), np.abs(mean_normal), "ocd_abs_mean", "CN", "OCD", "Mean Distribution")
        elif args.task:
            two_box(np.abs(var_unnormal), np.abs(var_normal), "task_abs_var", "CN", "AD", "VAR Distribution")
            two_box(np.abs(icc_unnormal), np.abs(icc_normal), "task_abs_icc", "CN", "AD", "ICC Distribution")
            two_box(np.abs(mean_unnormal), np.abs(mean_normal), "task_abs_mean", "CN", "AD", "Mean Distribution")
        else:
            print("no box plot drawing!")

    if args.aal_pattern:
        plot_mat_pattern(cfc_ave_normal, cfc_ave_unnormal, "CN", "AD", "aal")
    elif args.hcp_pattern:
        plot_mat_pattern(cfc_ave_normal, cfc_ave_unnormal, "Unaffected", "Schezophrannia", "hcp")
    elif args.ocd_pattern:
        plot_mat_pattern(cfc_ave_normal, cfc_ave_unnormal, "CN", "OCD", "ocd")
    elif args.task_pattern:
        # plot_mat_pattern_task(cfc_ave_a, cfc_ave_b, cfc_ave_c, cfc_ave_d, cfc_ave_e, cfc_ave_f, cfc_ave_g, cfc_ave_h, "a", "b", "c", "d", "e", "f", "g", "h", "task")
        plot_mat_pattern(cfc_ave_0bk, cfc_ave_2bk, "0bk", "2bk", "taskbk")
        plot_mat_pattern(cfc_ave_place, cfc_ave_body, "place", "body", "tasksub1")
        plot_mat_pattern(cfc_ave_face, cfc_ave_tools, "face", "tools", "tasksub2")
    else:
        print("no cfc pattern drawing!")

    if args.training_pattern:
        # training_cfc_all_unnormal, training_cfc_all_normal = plot_training_pattern(args.training_data_path, args.testing_data_path, args.o_data_path, args.to_data_path)
        training_cfc_all_unnormal, training_cfc_all_normal, step, walk = plot_training_pattern_newformat(args.training_data_path, args.testing_data_path, args.o_data_path, args.to_data_path)
        step = np.array(step)
        node_count = Counter(step)
        print(node_count)
        node_model = np.loadtxt(os.path.abspath('./data/Node_AAL{}.txt'.format(args.node_num)))
        node_plot = node_model
        for node_num in range(args.node_num):
            temp_un = node_count[node_num]
            node_plot[node_num][3] = temp_un
            node_plot[node_num][4] = temp_un

        node_count = node_count.most_common(30)
        node_plot[int(node_count[0][0])][3] = 0
        node_plot[int(node_count[0][0])][4] = 0
        # int(node_count[0][0])
        print(node_count)
        weight = np.zeros(args.node_num)
        for i in range(30):
            weight[int(node_count[i][0])] = node_count[i][1]

        walk_count = Counter(walk)
        tracj_garph = np.zeros((args.node_num, args.node_num))
        tracj_garph_nodirection = np.zeros((args.node_num, args.node_num))
        for walk in list(walk_count.elements()):
            walk = walk.split("\'")
            node1 = walk[1]
            node2 = walk[3]
            if int(node_count[0][0]) not in [int(node1), int(node2)]:
                tracj_garph[int(node1), int(node2)] = tracj_garph[int(node1), int(node2)] + 1
                tracj_garph_nodirection[int(node1), int(node2)] = tracj_garph_nodirection[int(node1), int(node2)] + 1
                tracj_garph_nodirection[int(node2), int(node1)] = tracj_garph_nodirection[int(node2), int(node1)] + 1

        walk_count = walk_count.most_common(50)
        print(walk_count)

        np.savetxt('./data/tracj_garph_delete.txt', tracj_garph, fmt='%f', delimiter=' ')
        np.savetxt('./data/tracj_garph_nodirection_delete.txt', tracj_garph_nodirection, fmt='%f', delimiter=' ')
        np.savetxt('./data/node_plot_delete.txt', node_plot, fmt='%f', delimiter=' ')
        training_cfc_all_normal, training_cfc_all_unnormal, sample_normal_num, sample_unnormal_num = matrix_cfc_plot_weight_train(args.cfc_data_path, args.processed_data_path, args.node_num, args.wavelets_num, weight)
        print(weight)
        print(weight.sum())
        training_cfc_all_unnormal = np.array(training_cfc_all_unnormal)
        training_cfc_all_normal = np.array(training_cfc_all_normal)
        training_cfc_ave_unnormal = np.mean(training_cfc_all_unnormal, axis=0)
        training_cfc_ave_normal = np.mean(training_cfc_all_normal, axis=0)
        plot_mat_pattern(training_cfc_ave_normal, training_cfc_ave_unnormal, "CN", "AD", "training_aal")
        var_normal, icc_normal = get_diag_value(training_cfc_all_normal)
        var_unnormal, icc_unnormal = get_diag_value(training_cfc_all_unnormal)
        # mean_normal = get_diag_mean(training_cfc_all_normal)
        # mean_unnormal = get_diag_mean(training_cfc_all_unnormal)
        name_path = os.path.split(args.to_data_path)
        two_box(np.abs(var_unnormal), np.abs(var_normal), "{}_abs_var_training_1202".format(name_path[1]), "CN", "AD", "VAR Distribution")
        two_box(np.abs(icc_unnormal), np.abs(icc_normal), "{}_abs_icc_training_1202".format(name_path[1]), "CN", "AD", "ICC Distribution")
        # two_box(np.abs(mean_unnormal), np.abs(mean_normal), "{}_abs_mean_training_1202".format(name_path[1]), "CN", "AD", "Mean Distribution")
        np.savetxt('./data/icc_unnormal.txt', icc_unnormal, fmt='%f', delimiter=' ')
        np.savetxt('./data/icc_normal.txt', icc_normal, fmt='%f', delimiter=' ')
    else:
        print("no training cfc drawing!")


# count_num_cn = 107
# count_num_ad = 72
# cfc_all_cn, cfc_all_ad = matrix_cfc_plot('D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\data\\AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5', 'D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\data\\AAL_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5')# AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5/aal_cn_ad_cfc_iter1000
#
#
# cfc_all_cn_node = np.zeros((90*count_num_cn, 10, 10))
# cfc_all_ad_node = np.zeros((90*count_num_ad, 10, 10))
# for sample_i in range(count_num_ad):
#     for node in range(90):
#         cfc_all_ad_node[node+90*sample_i] = cfc_all_ad[node, sample_i]
#
# for sample_i in range(count_num_cn):
#     for node in range(90):
#         cfc_all_cn_node[node+90*sample_i] = cfc_all_cn[node, sample_i]
# # tem_ave_cn, tem_con_cn = get_diag_value(cfc_all_cn_node)
# # tem_ave_ad, tem_con_ad = get_diag_value(cfc_all_ad_node)
# tem_ave_cn = get_diag_mean(cfc_all_cn_node)
# tem_ave_ad = get_diag_mean(cfc_all_ad_node)
# two_box(np.abs(tem_ave_ad), np.abs(tem_ave_cn), "abs_mean_12_45")
# # two_box(np.abs(tem_ave_ad), np.abs(tem_ave_cn), "abs_var_15")
# # two_box(np.abs(tem_con_ad), np.abs(tem_con_cn), "abs_con_15")
#
#
#
#
# # tem_arr_cn = np.zeros((14, 9, count_num_cn))
# # tem_arr_ad = np.zeros((14, 9, count_num_ad))
# #
# # for find_node in range(29, 43):
# #     tem_arr_ad[find_node-29] = get_diag_value(cfc_all_ad[find_node-1])
# #     tem_arr_cn[find_node-29] = get_diag_value(cfc_all_cn[find_node-1])
# #     two_box(tem_arr_ad[find_node-29], tem_arr_cn[find_node-29], find_node)
# #     two_box(np.abs(tem_arr_ad[find_node - 29]), np.abs(tem_arr_cn[find_node - 29]), "abs_{}".format(find_node))
#
#
#
# #
# # train_results = joblib.load("D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\IPMI2023\\IPMI-2023\\result.pkl")
# results = joblib.load(
#     "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\result.pkl")
# swq_attn = joblib.load(
#     "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\var\\swq_attn.pkl")
#
# # 1_5 test
# cn_cfc_ave_test = np.zeros((18540, 10, 10))
# ad_cfc_ave_test = np.zeros((8100, 10, 10))
# cn_num_count = 0
# ad_num_count = 0
# after_feature = []
# k = 0
# for i in range(26):
#     for j in range(len(swq_attn[60*i+59]['graph_seq'][1])):
#         after_feature.append(swq_attn[60*i+59]['graph_seq'][1][j])
#         for spea_i in range(len(after_feature[k])):
#             a = after_feature[k][spea_i].tolist()
#             cfc_matrix: np.array = np.array(a).astype('double')
#             if int(swq_attn[60*i+59]['graph_seq'][2][0]) == 0:
#                 cn_cfc_ave_test[cn_num_count] = cfc_matrix
#                 cn_num_count = cn_num_count + 1
#             else:
#                 ad_cfc_ave_test[ad_num_count] = cfc_matrix
#                 ad_num_count = ad_num_count + 1
#         k += 1
#
# # tem_ave_cn_train, tem_con_cn_train = get_diag_value(cn_cfc_ave_test)
# # tem_ave_ad_train, tem_con_ad_train = get_diag_value(ad_cfc_ave_test)
# # two_box(np.abs(tem_ave_ad_train), np.abs(tem_ave_cn_train), "abs_var_test_1_5")
# # two_box(np.abs(tem_con_ad_train), np.abs(tem_con_cn_train), "abs_con_test_1_5")
# tem_ave_cn_train = get_diag_mean(cn_cfc_ave_test)
# tem_ave_ad_train = get_diag_mean(ad_cfc_ave_test)
# two_box(np.abs(tem_ave_ad_train), np.abs(tem_ave_cn_train), "abs_mean_test_1_5")
#
# # 12_45 test
# cfc_all_ad_train = np.zeros((1320, 10, 10))# 440
# cfc_all_cn_train = np.zeros((1380, 10, 10))# 460
# cfc_ave_ad = np.zeros((10, 10))
# cfc_ave_cn = np.zeros((10, 10))
# test_num_ad = 0
# test_num_cn = 0
#
# for i in range(len(results)):
#     a = results[i]['enc_out'].tolist()
#     train_cfc: np.array = np.array(a).astype('double')
#     if int(results[i]['target']) == 1:
#         cfc_ave_ad = cfc_ave_ad + train_cfc[0][0]
#         cfc_all_ad_train[test_num_ad] = train_cfc[0][0]
#         test_num_ad = test_num_ad + 1
#     else:
#         cfc_ave_cn = cfc_ave_cn + train_cfc[0][0]
#         cfc_all_cn_train[test_num_cn] = train_cfc[0][0]
#         test_num_cn = test_num_cn + 1
#
# # 12_45 train_result test
# # for i in range(len(train_results)-len(results), len(train_results)):
# #     if (i-len(train_results)+len(results))%60>39:
# #         a = train_results[i]['gam_output'].tolist()
# #         train_cfc: np.array = np.array(a).astype('double')
# #         if int(results[i-len(train_results)+len(results)]['target']) == 1:
# #             cfc_ave_ad = cfc_ave_ad + train_cfc[0]
# #             cfc_all_ad_train[test_num_ad] = train_cfc[0]
# #             test_num_ad = test_num_ad + 1
# #         else:
# #             cfc_ave_cn = cfc_ave_cn + train_cfc[0]
# #             cfc_all_cn_train[test_num_cn] = train_cfc[0]
# #             test_num_cn = test_num_cn + 1
#
# # tem_ave_cn_train, tem_con_cn_train = get_diag_value(cfc_all_cn_train)
# # tem_ave_ad_train, tem_con_ad_train = get_diag_value(cfc_all_ad_train)
# # two_box(np.abs(tem_ave_ad_train), np.abs(tem_ave_cn_train), "abs_var_test_12_45")
# # two_box(np.abs(tem_con_ad_train), np.abs(tem_con_cn_train), "abs_con_test_12_45")
# tem_ave_cn_train = get_diag_mean(cfc_all_cn_train)
# tem_ave_ad_train = get_diag_mean(cfc_all_ad_train)
# two_box(np.abs(tem_ave_ad_train), np.abs(tem_ave_cn_train), "abs_mean_test_12_45")
#
# # cfc pattern
# # cfc_ave_ad = cfc_ave_ad/test_num_ad
# # cfc_ave_cn = cfc_ave_cn/test_num_cn
# #
# # for i in range(10):
# #     cfc_ave_ad[i, i] = 0
# #     cfc_ave_cn[i, i] = 0
# #
# # mins = []
# # maxes = []
# # mins.append(np.min(cfc_ave_ad))
# # mins.append(np.min(cfc_ave_cn))
# # maxes.append(np.max(cfc_ave_ad))
# # maxes.append(np.max(cfc_ave_cn))
# # vmin = np.min(mins)
# # vmax = np.max(maxes)
# # # vmin = np.min(np.min(cfc_ave_ad), np.min(cfc_ave_cn))
# # # vmax = np.max(np.max(cfc_ave_ad), np.max(cfc_ave_cn))
# # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(4, 2))
# # # fig.subplots_adjust(hspace=0.1)
# # xlabels = '1 2 3 4 5 6 7 8 9 10'.split()
# # ylabels = '1 2 3 4 5 6 7 8 9 10'.split()
# # ax = axs[0]
# # im = ax.imshow(cfc_ave_ad, vmin=vmin, vmax=vmax)#, cmap='Blues'
# # ax.set_title('AD_cfc_ave', fontdict={'family': 'Times New Roman', 'size': 12})
# # labels = ax.get_xticklabels() + ax.get_yticklabels()
# # [label.set_fontname('Times New Roman') for label in labels]
# # [label.set_fontsize(12) for label in labels]
# # ax = axs[1]
# # im = ax.imshow(cfc_ave_cn, vmin=vmin, vmax=vmax)#, cmap='Blues'
# # ax.set_title('CN_cfc_ave', fontdict={'family': 'Times New Roman', 'size': 12})
# # labels = ax.get_xticklabels() + ax.get_yticklabels()
# # [label.set_fontname('Times New Roman') for label in labels]
# # [label.set_fontsize(12) for label in labels]
# # plt.xticks(range(len(xlabels)), xlabels, fontproperties='Times New Roman', size=12)
# # plt.yticks(range(len(ylabels)), ylabels, fontproperties='Times New Roman', size=12)
# # cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
# # cbar = fig.colorbar(im, cax=cb_ax)
# # cbar_ax = cbar.ax
# # cbar_ax.tick_params(labelsize=12)
# # # cbar_ax.tick_params(which='major', direction='in', labelsize=12, length=7.5)
# # fig.savefig('D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\var\\cfc_corr_allave_12_45_test.svg',
# #                 format='svg', bbox_inches='tight', dpi=500)
# # plt.close('all')
