#-*- coding : utf-8 -*-
# coding: unicode_escape

import networkx as nx
import collections
from matplotlib import pyplot as plt
import operator
import math
import numpy as np
import random
from itertools import permutations 
from itertools import combinations 
import copy
import scipy.io as sio
import pandas as pd
from networkx.algorithms.cluster import triangles
import time
import subprocess
import csv
import codecs   
from scipy import stats
from params import *

#################################################
print("[Start]")
start_time = int(time.time())
time_stamp = start_time

# Update timestamp #
f = open(params_name,'r')
lines = f.readlines()
f.close()
for i in range(len(lines)):
    if 'time_stamp = ' in lines[i]:
        lines[i] = 'time_stamp = ' + str(start_time) + ' \\n'
f = open(params_name,'w')
for i in range(len(lines)):
    f.write(lines[i])
f.close()  

###################

ranking_matrix = []

for kk in range(len(alpha_pool)):
    
    print('Progress: ' + str(kk+1) + ' / ' + str(len(alpha_pool)))
    f = open(params_name,'r')
    lines = f.readlines()
    f.close()

    for i in range(len(lines)):
        if 'alpha = ' in lines[i]:
            lines[i] = 'alpha = ' + str(alpha_pool[kk]) + '\\n'
            
    f = open(params_name,'w')
    for i in range(len(lines)):
        f.write(lines[i])
    f.close()  
    
    ret = subprocess.run(command1,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=timeout_max)
    if ret.returncode == 0:
        print("Success - Forward Run")
    else:
        print("Fail - Forward Run")
    
    ret = subprocess.run(command2,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=timeout_max)
    if ret.returncode == 0:
        print("Success - EpiRank")
    else:
        print("Fail - EpiRank")
    print('Elapsed Time: ' + str(int(time.time() - start_time)) + 's.')
    
    filename_temp = 'output/' + str(time_stamp) + '_EpiRank_alpha_'+ str(alpha_pool[kk]) +'.csv'
    result = pd.read_csv(filename_temp, encoding="utf-8")
    values = [k for k in list(result.loc[:, '4'])]
    ranking_matrix.append(values)

rho, pval = stats.spearmanr(ranking_matrix,axis=1)
df = pd.DataFrame(rho)
df.to_csv('output/' + str(time_stamp) + '_SpearmanRho'+'.csv',encoding="utf-8-sig")
df = pd.DataFrame(pval)
df.to_csv('output/' + str(time_stamp) + '_SpearmanPval'+'.csv',encoding="utf-8-sig")

print("[End]")    