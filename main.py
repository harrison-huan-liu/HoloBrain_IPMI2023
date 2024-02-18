# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
# import os
import pandas as pd

def cfc_matrix(txtfile_path):
    # Use a breakpoint in the code line below to debug your script.
    corr_matrix = np.zeros((90,90))
    with open(txtfile_path,"r") as f:
        all_data = f.readlines()
        for i,line in enumerate(all_data):
            numbers = line.split()
            for j,element in enumerate(numbers):
                corr_matrix[i,j] = float(element)
    # print(corr_matrix)  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    effective_data = pd.read_excel('Effective_Data.xlsx',header=None)
    sample_name = pd.DataFrame(effective_data)
    sample_label = sample_name[0]
    sample_time = sample_name[1]
    label_first = ''
    for (label,time) in (sample_label,sample_time):
        txtfile_path = '.\Matrix\{label}_{time}'
        cfc_matrix(txtfile_path)
        print(txtfile_path)
    ### read all txt files
    # rootdir = os.path.join(".\Matrix")
    # for (dirpath,dirnames,filenames) in os.walk(rootdir):
    #     for filename in filenames:
    #         if os.path.splitext(filename)[1] == '.txt':
    #             txtfile_path = f'.\Matrix\{filename}'
    #             cfc_matrix(txtfile_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
