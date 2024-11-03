#!/bin/bash
yhrun -N 1 -c 8 -n 1 python train.py --desc "The data used is aal_cn_ad_cfc_iter1000, the value of the diagonal is 1, add node_attention, add gnnexplainer_reward, debug pattern, epochs 45, repetitions 60" --data_folder "./data/aal_cn_ad_cfc_iter1000" --node_attention --reward --debug --epochs 45 --repetitions 60