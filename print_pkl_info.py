import pickle
import sys
import networkx as nx
import torch
import numpy as np

if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    # file_path = '/Users/huan/Documents/HoloBrain_IPMI2023/data/AAL_CFC_CN_1_SMC_2_EMCI_3_LMCI_4_AD_5/sub-002S0295_aal.txt_2.pkl'
    file_path = '/Users/huan/Documents/HoloBrain_IPMI2023/IPMI-2023/1.pkl'

try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(data)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except pickle.UnpicklingError:
    print("Error: The file content is not a valid pickle format.")
except EOFError:
    print("Error: The file is incomplete or corrupted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
