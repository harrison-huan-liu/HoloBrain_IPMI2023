import numpy as np
import pickle
import os
from tqdm import tqdm, trange
from utils import sorted_aphanumeric
from dataset import GraphSeqDataset, GraphDataset
import pandas as pd
import warnings

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Generator Data.")

    parser.add_argument(
        '--original_subjectinfo_data_path',
        nargs="?",
        default="./data/AAL_90 DATA/subject_info.csv",
        help='The subject information of original data',
    )

    parser.add_argument(
        '--original_data_path',
        nargs="?",
        default="./data/AAL_90 DATA/AAL90",# "./data/Schezophrannia_Diana_Jefferies",
        help='The original data',
    )

    parser.add_argument(
        '--processed_data_path',
        nargs="?",
        default="./data/AAL_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5",
        help='The processed data in python format',
    )

    parser.add_argument(
        '--cfc_data_path',
        nargs="?",
        default="./data/AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5",
        help='The calculated cfc data',
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
        help="process the hcp data",
    )

    parser.add_argument(
        "--ocd",
        action='store_true',
        default=False,
        help="process the ocd data",
    )

    parser.add_argument(
        "--task",
        action='store_true',
        default=False,
        help="process the task data",
    )

    return parser.parse_args()


def check_data(data_path):
    files = sorted_aphanumeric(
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if file.startswith('TimeSeries')
    )
    for file in tqdm(files, 'checking data'):
        data = np.loadtxt(file, delimiter=',')
        assert not (np.isnan(data).any() or np.isinf(data).any())
    print('check sucessfully!')


def task_data(data_path, label_path):
    files = [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if file.startswith('TimeSeries')
    ]
    labels = np.loadtxt(label_path, delimiter=',')
    ans = []
    for file in tqdm(files, 'generating data'):
        data = np.loadtxt(file, delimiter=',')
        for label in range(8):
            mat = data[..., labels == label]
            ans.append([mat, label])
    return ans


def aal_data(info_path, data_path):
    ans = []
    df = pd.read_csv(info_path)
    df = df[df["DX"].isin(["CN", "SMC", "EMCI", "LMCI", "AD"])]
    for name, dx in tqdm(zip(df['subject_id'], df['DX']), total=len(df)):
        # label = 0 if dx in ['CN'] else 1
        if dx in ['CN']:
            label = 1
        elif dx in ['SMC']:
            label = 2
        elif dx in ['EMCI']:
            label = 3
        elif dx in ['LMCI']:
            label = 4
        else:
            label = 5
        filename = 'sub-' + name.replace('_', '') + '_aal.txt'
        filepath = os.path.join(data_path, filename)
        if pd.isna(name) or pd.isna(dx) or not os.path.exists(filepath):
            warnings.warn(f'{name} does not exist')
            continue
        mat = np.loadtxt(filepath, dtype=np.float32)[..., :90].T
        ans.append([mat, label])
    return ans


def hcp_data(info_path, data_path):
    ans = []
    for name in ["Cleaned_Unaffected2", "Cleaned_Nonconverters2", "Cleaned_Converters2"]:
        diff_path = os.path.join(data_path, name)
        all_id = os.listdir(diff_path)
        for id in tqdm(all_id, total=len(all_id)):
            filepath = os.path.join(diff_path, id)
            mat = np.loadtxt(filepath, dtype=np.float32)[..., :116].T
            if name in ["Cleaned_Unaffected2"]:
                label = 0
            else:
                label = 1
            ans.append([mat, label])
    return ans


def ocd_data(info_path, data_path):
    ans = []
    for name in ["OCD", "CN"]:
        diff_path = os.path.join(data_path, name)
        all_id = os.listdir(diff_path)
        for id in tqdm(all_id, total=len(all_id)):
            filepath = os.path.join(diff_path, id)
            mat = np.loadtxt(id, dtype=np.float32)[..., :116].T
            if name in ["CN"]:
                label = 0
            else:
                label = 1
            ans.append([mat, label])
    return ans


def save_data(datas, path):
    os.makedirs(path, exist_ok=True)
    for i, data in tqdm(enumerate(datas, 1), 'saving data', total=len(datas)):
        with open(os.path.join(path, f'{i}.pkl'), 'wb') as f:
            pickle.dump(data, f)


def save_data_singledata(datas, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f'result.pkl'), 'wb') as f:
        pickle.dump(datas, f)


def graph_seq_cfc_data(
    data_path,
    save_path,
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
    dataset = GraphSeqDataset(
        data_path,
        window=window,
        step=step,
        padding=padding,
        ratio=ratio,
        wavelets_num=wavelets_num,
        beta=beta,
        gamma=gamma,
        max_iter=max_iter,
        min_err=min_err,
        node_select=node_select,
    )
    os.makedirs(save_path, exist_ok=True)
    for i in trange(len(dataset)):
        graphs, cfcs, labels = dataset[i]
        data = {'graph': graphs, 'cfc': cfcs, 'label': int(labels[0])}
        with open(os.path.join(save_path, f'{i}.pkl'), 'wb') as f:
            pickle.dump(data, f)


def graph_cfc_data(
    data_path,
    save_path,
    ratio,
    wavelets_num,
    beta,
    gamma,
    max_iter,
    min_err,
    node_select,
):
    dataset = GraphDataset(
        data_path,
        ratio=ratio,
        wavelets_num=wavelets_num,
        beta=beta,
        gamma=gamma,
        max_iter=max_iter,
        min_err=min_err,
        node_select=node_select,
    )
    os.makedirs(save_path, exist_ok=True)
    count_ad_category = 0
    count_cn_category = 0
    for i in trange(len(dataset)):
        graphs, cfcs, labels = dataset[i]
        if labels != 3:
            data = {'graph': graphs, 'cfc': cfcs, 'label': int(labels[0]>3)}
            with open(os.path.join(save_path, f'{i}.pkl'), 'wb') as f:
                pickle.dump(data, f)
            if int(labels[0]>3)==1:
                count_ad_category+=1
            else:
                count_cn_category+=1
    print("AD:", count_ad_category, "\nCN:", count_cn_category)


if __name__ == '__main__':
    args = parameter_parser()

    if args.aal:
        # AAL
        save_data(
            aal_data(
                args.original_subjectinfo_data_path,
                args.original_data_path,
            ),
            args.processed_data_path,
        )
    elif args.hcp:
        save_data(
            hcp_data(
                args.original_subjectinfo_data_path,
                args.original_data_path,
            ),
            args.processed_data_path,
        )
    elif args.ocd:
        save_data(
            ocd_data(
                args.original_subjectinfo_data_path,
                args.original_data_path,
            ),
            args.processed_data_path,
        )
    elif args.task:
        print("select part of task data from the data folder")


    graph_seq_cfc_data(
        data_path=args.processed_data_path,
        save_path=args.cfc_data_path,
        window=40,
        step=15,
        padding=True,
        ratio=0.4,
        wavelets_num=10,
        beta=1,
        gamma=0.005,
        max_iter=1000,
        min_err=0.0001,
        node_select=10,
    )

    # scan1_path = r'D:\Projects\Pycharm\数据\FullData_Oct26_true\Scan1'
    # scan1_label_path = (
    #     r'D:\Projects\Pycharm\数据\FullData_Oct26_true\task_scan1_label_norest.csv'
    # )
    # scan2_path = r'D:\Projects\Pycharm\数据\FullData_Oct26_true\Scan2'
    # scan2_label_path = (
    #     r'D:\Projects\Pycharm\数据\FullData_Oct26_true\task_scan2_label_norest.csv'
    # )
    # # check_data(scan1_path)
    # # check_data(scan2_path)
    # datas = task_data(scan1_path, scan1_label_path) + task_data(
    #     scan2_path, scan2_label_path
    # )
    # save_data(datas, 'data/task')
    # print(datas)

    # graph_cfc_data(
    #     data_path='data/task',
    #     save_path='data/cfc',
    #     window=20,
    #     step=6,
    #     padding=True,
    #     ratio=0.4,
    #     wavelets_num=10,
    #     beta=1,
    #     gamma=0.005,
    #     max_iter=1,
    #     min_err=0.0001,
    #     node_select=10,
    # )

    # graph_cfc_data(
    #     data_path='data/AAL_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5',
    #     save_path='data/AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5',
    #     ratio=0.4,
    #     wavelets_num=10,
    #     beta=1,
    #     gamma=0.005,
    #     max_iter=1000,
    #     min_err=0.0001,
    #     node_select=10,
    # )
