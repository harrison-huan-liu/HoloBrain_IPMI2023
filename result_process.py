import numpy as np
import os
import pickle
import torchvision
import matplotlib.pyplot as plt
from numpy import linalg

ad_cfc = np.zeros((10,10))
ad_num = 274
ad_num_count = 0
ad_sigma = np.zeros((ad_num, 10))
cn_cfc = np.zeros((10,10))
cn_num = 1042
cn_num_count = 0
cn_sigma = np.zeros((cn_num, 10))
for filename in os.listdir('./data/aal_cn_ad_cfc_iter1000'):
    abs_path = os.path.abspath('./data/aal_cn_ad_cfc_iter1000')
    complete_path = os.path.join(abs_path, filename)
    cfc_label = open(complete_path, 'rb')
    cfc = pickle.load(cfc_label)
    cfc_label.close()
    for spea_i in range(0, len(cfc['cfc'])):
        a = cfc['cfc'][spea_i].tolist()
        cfc_matrix:np.array = np.array(a).astype('double')
        gnn_cfc = cfc_matrix[28]
        if cfc['label'] == 0:
            cn_cfc = cn_cfc + gnn_cfc
            u, cn_sigma[cn_num_count, :], vt = linalg.svd(gnn_cfc)
            cn_num_count = cn_num_count + 1
        else:
            ad_cfc = ad_cfc + gnn_cfc
            u, ad_sigma[ad_num_count, :], vt = linalg.svd(gnn_cfc)
            ad_num_count = ad_num_count + 1

ad_cfc = ad_cfc/ad_num
cn_cfc = cn_cfc/cn_num
colors = ['DeepSkyBlue','white','Crimson']

import matplotlib as mpl
colormap = mpl.colors.ListedColormap(colors)
for i in range(0, 10):
    ad_cfc[i,i] = 0
    cn_cfc[i, i] = 0
for j in range(0, 10):
    for k in range(0,10):
        if ad_cfc[j,k]>0:
            ad_cfc[j,k]=1
        elif ad_cfc[j,k]<0:
            ad_cfc[j,k]=-1
        else:
            ad_cfc[j, k] = 0
        if cn_cfc[j,k]>0:
            cn_cfc[j,k] = 1
        elif cn_cfc[j,k]<0:
            cn_cfc[j,k] = -1
        else:
            cn_cfc[j, k] = 0
ax = plt.matshow(ad_cfc, cmap = colormap)
plt.colorbar(ax.colorbar, fraction=0.025)
plt.title('ad_cfc_matrix')
ax = plt.matshow(cn_cfc, cmap = colormap)
plt.colorbar(ax.colorbar, fraction=0.025)
plt.title('cn_cfc_matrix')


import seaborn as sns
from scipy.stats import norm, exponpow, exponnorm, expon, beta
sns.set_palette("hls") #设置所有图的颜色，使用hls色彩空间
sns.distplot(ad_sigma, bins=500, kde_kws={"color":"red", "lw":3 }, hist_kws={ "color": "r" }, label="AD", norm_hist=False)#
sns.distplot(cn_sigma, bins=500, kde_kws={"color":"green", "lw":3 }, hist_kws={ "color": "g" }, label="CN", norm_hist=False)#
# sns.distplot(sigma, bins=100, fit=exponpow, kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "b" })
# sns.distplot(sigma, bins=100, fit=exponnorm, kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "b" })
# sns.distplot(sigma, bins=100, fit=expon, kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "b" })
plt.show()

count_cn_num = 0
for filename in os.listdir('./data/aal_cn_ad_cfc_iter1000'):
    abs_path = os.path.abspath('./data/aal_cn_ad_cfc_iter1000')
    complete_path = os.path.join(abs_path, filename)
    cfc_label = open(complete_path, 'rb')
    cfc = pickle.load(cfc_label)
    if cfc['label']==0:
        count_cn_num = count_cn_num + 1
    cfc_label.close()




# # fig1 = plt.figure()
# plt.matshow(cfc['graph'])
# plt.title('Graph')
# plt.savefig('./data/Graph.svg', format='svg', dpi=500)
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

# colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'lime', 'pink')
for i in range(0, 10):
    ad_cfc[i,i] = 0
# _range = np.max(cfc_matrix[N_i]) - np.min(cfc_matrix[N_i])
# cfc_matrix[N_i] = (cfc_matrix[N_i] - np.min(cfc_matrix[N_i])) / _range
# fig4 = plt.figure()
ax = plt.matshow(ad_cfc, cmap='jet')
plt.colorbar(ax.colorbar, fraction=0.025)
plt.title('cfc_matrix')
plt.savefig('./data/ad_cfc.svg', format='svg', dpi=500)



# the distribution of cfc_matrix
# from numpy import linalg
# node_num = 90
# sigma = np.zeros((node_num, 10))
# for N_i in range(node_num):
#     u, sigma[N_i, :], vt = linalg.svd(cfc_matrix[N_i])

plt.hist(sigma)
plt.legend(list("0123456789"))

# import scipy.io as scio
# Pdfile = '../Pd.mat'
# Pd = scio.loadmat(Pdfile)
#
# sigma_hub = np.zeros((134, 10))
# sigma_nonhub = np.zeros((134, 10))
# hub_num = len(Pd['Pd'][0])
#
# hub_i = 0
# nhub_i = 0
# for i in range(hub_num):
#     if Pd['Pd'][0][i] == 1:
#         sigma_hub[hub_i, :] = sigma[i, :]
#         hub_i = hub_i + 1
#     else:
#         sigma_nonhub[nhub_i, :] = sigma[i, :]
#         nhub_i = nhub_i + 1
#
# plt.hist(sigma_hub)
# plt.legend(list("0123456789"))
# plt.hist(sigma_nonhub)
# plt.legend(list("0123456789"))


import seaborn as sns
from scipy.stats import norm, exponpow, exponnorm, expon, beta
sns.set_palette("hls") #设置所有图的颜色，使用hls色彩空间
sns.distplot(ad_sigma, bins=100, fit=norm, kde_kws={"color":"red", "lw":3 }, hist_kws={ "color": "r" }, label="AD")# kde=False
sns.distplot(cn_sigma, bins=100, fit=norm, kde_kws={"color":"green", "lw":3 }, hist_kws={ "color": "g" },label="CN")# kde=False
# sns.distplot(sigma, bins=100, fit=exponpow, kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "b" })
# sns.distplot(sigma, bins=100, fit=exponnorm, kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "b" })
# sns.distplot(sigma, bins=100, fit=expon, kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "b" })
plt.show()

color = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'lime', 'pink')
plt.hist(ad_sigma, bins=50, color=color)
plt.legend(list("0123456789"))
