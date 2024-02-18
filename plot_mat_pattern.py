from wavelets_test import plot_matfig, mat_to_ones
import joblib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman', size=18, weight='bold')
import os
import pickle

# cfc_ad = joblib.load(
#     "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\var\\cfc_ad.pkl")
# cfc_cn = joblib.load(
#     "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\var\\cfc_cn.pkl")
# wavelets_corr_ad = joblib.load(
#     "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\var\\wavelets_corr_ad.pkl")
# wavelets_corr_cn = joblib.load(
#     "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\var\\wavelets_corr_cn.pkl")
# bold_diff_ad = joblib.load(
#     "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\var\\bold_diff_ad.pkl")
# bold_diff_cn = joblib.load(
#     "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\var\\bold_diff_cn.pkl")
# train_steps = joblib.load(
#     "D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\var\\train_steps.pkl")

# sign_node = [29, 30, 37, 38, 41, 42]
# colors = ['DeepSkyBlue', 'white', 'Crimson']
# colormap = mpl.colors.ListedColormap(colors)
#
# for plot_node in sign_node:
#     for i in range(0, 10):
#         cfc_ad[plot_node - 1, i, i] = 0
#         cfc_cn[plot_node - 1, i, i] = 0
#         wavelets_corr_ad[plot_node - 1, i, i] = 0
#         wavelets_corr_cn[plot_node - 1, i, i] = 0
#         bold_diff_ad[plot_node - 1, i, i] = 0
#         bold_diff_cn[plot_node - 1, i, i] = 0
#
# plot_matfig(3, 4, sign_node, cfc_ad, cfc_cn, 'viridis', 'cfc', 'viridis')
# plot_matfig(3, 4, sign_node, wavelets_corr_ad, wavelets_corr_cn, 'viridis', 'wave', 'viridis')
# plot_matfig(3, 4, sign_node, bold_diff_ad, bold_diff_cn, 'viridis', 'bold_diff', 'viridis')
# plt.close("all")
#
# for plot_node in sign_node:
#     cfc_ad[plot_node - 1] = mat_to_ones(cfc_ad[plot_node - 1])
#     cfc_cn[plot_node - 1] = mat_to_ones(cfc_cn[plot_node - 1])
#     wavelets_corr_ad[plot_node - 1] = mat_to_ones(wavelets_corr_ad[plot_node - 1])
#     wavelets_corr_cn[plot_node - 1] = mat_to_ones(wavelets_corr_cn[plot_node - 1])
#     bold_diff_ad[plot_node - 1] = mat_to_ones(bold_diff_ad[plot_node - 1])
#     bold_diff_cn[plot_node - 1] = mat_to_ones(bold_diff_cn[plot_node - 1])
#
# plot_matfig(3, 4, sign_node, cfc_ad, cfc_cn, colormap, 'cfc', 'two_color')
# plot_matfig(3, 4, sign_node, wavelets_corr_ad, wavelets_corr_cn, colormap, 'wave', 'two_color')
# plot_matfig(3, 4, sign_node, bold_diff_ad, bold_diff_cn, colormap, 'blod_diff', 'two_color')
# plt.close("all")

def isValidIndex(x, n):
    return (x >= 0 and x < n)
    # 每一行的每个值的数组下标的差都一样，

def get_diag_value(cfc):
    sample_num = len(cfc)
    rows = cols = len(cfc[0])
    tem_arr = np.zeros((9, 9*sample_num))  # 用来记录数组值
    for sample in range(sample_num):
        for i in range(0, cols-1):  # 共输出 cols * 2 - 1 行
            diff = cols - i - 1  # 每一行的差
            for j in range(cols):  # 数组中每一个值的下标范围是0到cols
                k = j - diff  # 通过一个下标值计算另一个下标值
                if isValidIndex(k, rows):  # 剩下就是判断这些下标值是否满足当前的情况， 这一步不怎么好理解
                    tem_arr[i, k+sample*9] = cfc[sample, k, j]
                    # print(cfc[k, j], ' ', end='')

    return tem_arr


def two_box(data_ad, data_cn, find_node):
    labels = ["AD", "CN"]
    # 三个箱型图的颜色 RGB （均为0~1的数据）
    colors = [(202 / 255., 96 / 255., 17 / 255.), (255 / 255., 217 / 255., 102 / 255.)] #, (137 / 255., 128 / 255., 68 / 255.)
    for i in range(9):
        data_ad_cn = [data_ad[i, :][data_ad[i, :]!=0], data_cn[i, :][data_cn[i, :]!=0]]
        # 绘制箱型图
        # patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
        bplot = plt.boxplot(data_ad_cn, patch_artist=True, labels=labels, positions=(i+1.2, i+1.6), widths=0.3, showmeans=True, showfliers=False)
        # 将三个箱分别上色
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    x_position = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_position_fmt = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)

    plt.ylabel('CFC value')
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    plt.legend(bplot['boxes'], labels, bbox_to_anchor=(1, 1))  # 绘制表示框，右下角绘制
    plt.title("node {}".format(find_node))
    plt.savefig(fname="D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\box_plot\\pic_{}_new.svg".format(find_node), format='svg', bbox_inches='tight', dpi=500)
    plt.close("all")
    # plt.show()


def matrix_cfc_plot(count_num_cn, count_num_ad, cfc_path, data_path):
    AD_num = 0
    CN_num = 0
    cfc_ave_cn = np.zeros((10, 10))
    cfc_ave_ad = np.zeros((10, 10))
    cfc_all_cn = np.zeros((116, count_num_cn, 10, 10))# 90
    cfc_all_ad = np.zeros((116, count_num_ad, 10, 10))# 90
    for filename, filename_o in zip(os.listdir(cfc_path), os.listdir(data_path)):
        cfc_mat = np.zeros((116, 10, 10))# 90
        abs_path = os.path.abspath(cfc_path)
        complete_path = os.path.join(abs_path, filename)
        cfc_label = open(complete_path, 'rb')
        cfc = pickle.load(cfc_label)
        # cfc_mat = joblib.load(complete_path)
        print(len(cfc['cfc']))
        for i in range(len(cfc['cfc'])):
            a = cfc['cfc'][i].tolist()
            cfc_matrix: np.array = np.array(a).astype('double')
            cfc_mat = cfc_mat + cfc_matrix

        cfc_mat = cfc_mat / len(cfc['cfc'])
        # cfc = open(complete_path, 'rb')
        # cfc_mat = pickle.load(cfc)
        # abs_path_label = os.path.abspath(data_path)
        # complete_path_label = os.path.join(abs_path_label, filename_o)
        # cfc_label = open(complete_path_label, 'rb')
        # BOLD_window, label = pickle.load(cfc_label)
        if cfc['label'] == 0 and CN_num<count_num_cn:
            for all_node_num in range(116):# 90
                cfc_all_cn[all_node_num, CN_num] = cfc_mat[all_node_num]
                cfc_ave_cn = cfc_ave_cn + cfc_mat[all_node_num]
            CN_num += 1
        elif cfc['label'] == 1 and AD_num<count_num_ad:
            for all_node_num in range(116):# 90
                cfc_all_ad[all_node_num, AD_num] = cfc_mat[all_node_num]
                cfc_ave_ad = cfc_ave_ad + cfc_mat[all_node_num]
            AD_num += 1
        else:
            print("AD or CN data is enough! Sample: ", filename)

    return cfc_all_cn, cfc_all_ad, cfc_ave_cn, cfc_ave_ad

count_num_cn = 72
count_num_ad = 72
cfc_all_cn, cfc_all_ad, cfc_ave_cn, cfc_ave_ad = matrix_cfc_plot(count_num_cn, count_num_ad, 'D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\data\\hcp_cfc', 'D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\data\\hcp_cfc')# AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5/aal_cn_ad_cfc_iter1000/hcp_cfc


tem_arr_cn = np.zeros((14, 9, 9*count_num_cn))
tem_arr_ad = np.zeros((14, 9, 9*count_num_ad))

tem_ave_cn = np.zeros((9, 810*count_num_cn))
tem_ave_ad = np.zeros((9, 810*count_num_ad))
for node in range(90):
    tem_ave_cn[:, node*9*count_num_cn:(node+1)*9*count_num_cn] = get_diag_value(cfc_all_cn[node])
    tem_ave_ad[:, node*9*count_num_ad:(node+1)*9*count_num_ad] = get_diag_value(cfc_all_ad[node])
two_box(np.abs(tem_ave_ad), np.abs(tem_ave_cn), "abs_all")

for find_node in range(29, 43):
    tem_arr_ad[find_node-29] = get_diag_value(cfc_all_ad[find_node-1])
    tem_arr_cn[find_node-29] = get_diag_value(cfc_all_cn[find_node-1])
    two_box(tem_arr_ad[find_node-29], tem_arr_cn[find_node-29], find_node)
    two_box(np.abs(tem_arr_ad[find_node - 29]), np.abs(tem_arr_cn[find_node - 29]), "abs_{}".format(find_node))

cfc_ave_ad = cfc_ave_ad/(90*count_num_ad)
cfc_ave_cn = cfc_ave_cn/(90*count_num_cn)

for i in range(10):
    cfc_ave_ad[i, i] = 0
    cfc_ave_cn[i, i] = 0

train_results = joblib.load("D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\IPMI2023\\IPMI-2023\\result.pkl")
results = joblib.load("D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\result.pkl")
cfc_ave_ad = np.zeros((10, 10))
cfc_ave_cn = np.zeros((10, 10))
test_num_ad = 0
test_num_cn = 0
for i in range(len(train_results)-len(results)):
    if i%45>29:
        a = train_results[i]['gam_output'].tolist()
        train_cfc: np.array = np.array(a).astype('double')
        cfc_ave_ad = cfc_ave_ad + train_cfc[0]
        test_num_ad = test_num_ad + 1
        cfc_ave_cn = cfc_ave_cn + train_cfc[0]
        test_num_cn = test_num_cn + 1
        # if int(results[i-len(train_results)+len(results)]['target']) == 1:
        #     cfc_ave_ad = cfc_ave_ad + train_cfc[0]
        #     test_num_ad = test_num_ad + 1
        # else:
        #     cfc_ave_cn = cfc_ave_cn + train_cfc[0]
        #     test_num_cn = test_num_cn + 1

# for i in range(len(train_results)-len(results), len(train_results)):
#     if (i-len(train_results)+len(results))%60>39:
#         a = train_results[i]['gam_output'].tolist()
#         train_cfc: np.array = np.array(a).astype('double')
#         if int(results[i-len(train_results)+len(results)]['target']) == 1:
#             cfc_ave_ad = cfc_ave_ad + train_cfc[0]
#             test_num_ad = test_num_ad + 1
#         else:
#             cfc_ave_cn = cfc_ave_cn + train_cfc[0]
#             test_num_cn = test_num_cn + 1


cfc_ave_ad = cfc_ave_ad/test_num_ad
cfc_ave_cn = cfc_ave_cn/test_num_cn

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
im = ax.imshow(cfc_ave_ad, vmin=vmin, vmax=vmax)#, cmap='Blues'
ax.set_title('AD_cfc_ave', fontdict={'family': 'Times New Roman', 'size': 12})
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(12) for label in labels]
ax = axs[1]
im = ax.imshow(cfc_ave_cn, vmin=vmin, vmax=vmax)#, cmap='Blues'
ax.set_title('CN_cfc_ave', fontdict={'family': 'Times New Roman', 'size': 12})
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
fig.savefig('D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\var\\cfc_corr_allave_12_45_after.svg',
                format='svg', bbox_inches='tight', dpi=500)
plt.close('all')
