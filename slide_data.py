import os
import math

all_file = os.listdir("./data/hcp")
each_num = math.ceil(len(all_file)/10)

for i in range(start_num, end_num):
    '{}.pkl'.format(i)