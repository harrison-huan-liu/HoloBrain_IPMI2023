import os
import pickle
import matplotlib.pyplot as plt
import joblib
import numpy as np

k=0
count_cn_num = 0
count_ad_num = 0
power_all_cn = np.zeros((10, 75*90))
power_all_ad = np.zeros((10, 27*90))
power_ave = np.zeros((102, 90, 10))
for filename in os.listdir('D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\results\\var\\power_iterthous'):
    abs_path = os.path.abspath('D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\results\\var\\power_iterthous')
    complete_path = os.path.join(abs_path, filename)
    power_iterthous = joblib.load(complete_path)
    # cfc = open(complete_path, 'rb')
    # cfc_mat = pickle.load(cfc)
    # power_ave[k] = np.mean(power_iterthous, axis=1)
    power_ave[k] = power_iterthous[:, 40, :]
    abs_path_label = os.path.abspath(
        'D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\data\\aal_adcn')
    complete_path_label = os.path.join(abs_path_label, filename)
    cfc_label = open(complete_path_label, 'rb')
    BOLD_window, label = pickle.load(cfc_label)
    if label == 0:
        power_all_cn[:, count_cn_num*90:count_cn_num*90+90] = power_ave[k].T
        count_cn_num = count_cn_num + 1
    else:
        power_all_ad[:, count_ad_num*90:count_ad_num*90+90] = power_ave[k].T
        count_ad_num = count_ad_num + 1
    k = k + 1

labels = ["AD", "CN"]
colors = [(202 / 255., 96 / 255., 17 / 255.), (255 / 255., 217 / 255., 102 / 255.)] #, (137 / 255., 128 / 255., 68 / 255.)
for i in range(10):
    data_ad_cn = [power_all_cn[i], power_all_ad[i]]
    bplot = plt.boxplot(data_ad_cn, patch_artist=True, labels=labels, positions=(i+1.2, i+1.6), widths=0.3, showmeans=True, showfliers=False)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    x_position = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x_position_fmt = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)

    plt.ylabel('Power')
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    plt.legend(bplot['boxes'], labels, bbox_to_anchor=(1, 1))  # 绘制表示框，右下角绘制
# x = np.arange(1, 141)
# # fig, axs = plt.subplots(5, 2)
# for i in range(521, 530):
#     plt.subplot(i)
#     plt.plot(x, power_iterthous[28, :, i-520])
