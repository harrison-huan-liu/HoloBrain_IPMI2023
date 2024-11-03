import csv
from pandas import array
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman', size=12, weight='bold')


def loadCSVfile1():
    list_file = []
    with open('D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\outputpredictions_2022-10-27_16-08-38_seq.csv','rb') as csv_file:
        all_lines=csv.reader(csv_file)
        for one_line in all_lines:
            list_file.append(one_line)
    list_file.remove(list_file[0])
    arr_file = array(list_file)
    label = arr_file[:, 0]
    data = arr_file[:, 1:]
    return data, label


def loadCSVfile2():
    tmp = np.loadtxt("D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\outputpredictions_2022-10-27_16-08-38_seq.csv", dtype=np.str, delimiter="\t")
    data = tmp[1:].astype(np.float)#加载数据部分
    label = tmp[1:].astype(np.float)#加载类别标签部分
    return data, label #返回array类型的数据


def loadCSVfile3():
    data = np.loadtxt(open("D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\outputpredictions_2022-10-27_16-08-38_seq.csv","rb"), delimiter="\t")
    return data


def counter(arr):
    return Counter(arr).most_common(90)  # 返回出现频率最高的两个数


with open('D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\2022-10-27_16-08-38trainsteps_2022-10-27_16-08-38_seq.json') as user_file:
  parsed_json = json.load(user_file)

print(parsed_json[0])
train_steps = []
for i in range(len(parsed_json)):
    train_steps.append(parsed_json[i]['train_steps'])

train_steps_array = np.array(train_steps)

for n in range(18):
    train_steps_vec = train_steps_array[:,n*10:10*(n+1)].ravel()

    # plt.hist(train_steps_vec, bins=np.arange(90))
    # plt.show()

    a = counter(train_steps_vec)

    x = []
    height = []
    for k in range(90):
        x.append(int(a[k][0])+1)
        height.append(a[k][1]/len(train_steps_vec))

    plt.figure()
    plt.bar(x, height, color='green', edgecolor='green', alpha=0.6, linewidth=2) #, tick_label=

    annotation_x=[]
    for j in range(10):
        annotation_x.append(int(a[j][0])+1)
        if int(a[j][0])<=9:
            plt.text(int(a[j][0])+0.3, a[j][1]/len(train_steps_vec)+0.0025, int(a[j][0])+1, fontsize=10, fontweight='bold', color='red')
        else:
            plt.text(int(a[j][0]) - 0.6, a[j][1] / len(train_steps_vec) + 0.0025, int(a[j][0]) + 1, fontsize=10, fontweight='bold',
                     color='red')


    plt.xlabel('Node', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.show()
    plt.savefig('D:\\research\\Nonliner Dimensional Reduction\\4_GNN_SPD\\CFC matrix\\var\\node_frequency_{}.svg'.format((18-n)*10),
                    format='svg', bbox_inches='tight', dpi=500)
    plt.close("all")
