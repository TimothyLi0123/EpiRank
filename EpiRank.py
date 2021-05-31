import csv
import numpy.matlib 
import numpy as np
import pandas as pd
from params import *

city_info = pd.read_csv(filename_py, encoding="utf-8")
popu = list(city_info.iloc[:,3])[:-1]
popu_flag = [int(k > popu_cutoff) for k in popu]
popu_index = [i for i,j in enumerate(popu_flag) if j == 1]
n_city = len(popu_index)

##########################################

data = []
f = open('output/' + str(time_stamp) + '_Matrix_I' +'.csv','r')
csv_reader = csv.reader(f)

count = -1
for row in csv_reader:
    if row[0] == '':
        continue
    count = count + 1
    if count == len(popu_flag):
        continue
    if popu_flag[count] == 0:
        continue
    data.append([float(k) for k in row[1:]])
    
data = np.matrix(data)
U1 = data[:,popu_index]


data = []
f = open('output/' + str(time_stamp) + '_Matrix_R' +'.csv','r')
csv_reader = csv.reader(f)

count = -1
for row in csv_reader:
    if row[0] == '':
        continue
    count = count + 1
    if count == len(popu_flag):
        continue
    if popu_flag[count] == 0:
        continue
    data.append([float(k) for k in row[1:]])
    
data = np.matrix(data)
U2 = data[:,popu_index]

U = U1 + U2

###########################################

I = np.matlib.eye(len(U))
DiagU = np.diag(np.diag(U))
P_minus = np.power(np.array([popu[k] for k in popu_index],dtype=np.float),-1)
One = np.matlib.ones((len(U),1))

F = np.multiply(U,np.matlib.ones((len(U),1))*np.power(sum(U.T)-np.diag(U),-1))
F = np.multiply(U,np.matlib.ones((len(U),1))*np.power(sum(U.T),-1))
DiagF = np.diag(np.diag(F))

H = (1-alpha)*(I - alpha*(F - DiagF))**-1*DiagU*One #np.diag(P_minus)*One

# Write results #
output_H = [list(city_info.iloc[popu_index[kkk],[1,3]]) + list([city_info.iloc[popu_index[kkk],19][:-1]]) + [float(H[kkk])] + [kkk] for kkk in range(len(popu_index))]
from operator import itemgetter, attrgetter
output_H = sorted(output_H, key = itemgetter(3), reverse=True)
df = pd.DataFrame(output_H)
df.to_csv('output/' + str(time_stamp) + '_EpiRank_alpha_'+ str(alpha) +'.csv',encoding="utf-8-sig")

###########################################

# Visualization #

#from pyecharts import Geo

#names_chn = list(city_info.keys()[0:-1])
#keys = list(city_info.iloc[:,1])
#keys = [keys[k] for k in popu_index]
#keys.reverse()

#filename_result = 'output/' + str(time_stamp) + '_EpiRank_alpha_'+ str(alpha) +'.csv'
#result = pd.read_csv(filename_result, encoding="utf-8")
#values = [max(round(k,2)-float(max(H)*visual_cutoff),0) for k in list(result.loc[:, '3'])]
#values.reverse()

#geo = Geo(title_1, title_2, title_color="#2F4F4F",title_pos="left", width=1200, height=600,background_color='#F0F8FF')

#def label_formatter(params):
#    return params.data.name

#legend = "EpiRank"
#for i in range(len(keys)):
#    geo.add(legend, [keys[i]], [values[i]], visual_range=[0,max(values)], type='effectScatter',visual_text_color="#2F4F4F", symbol_size=15*(values[i]/max(values))**0.8,is_visualmap=True, is_roam=True, tooltip_tragger='map',geo_normal_color="#6E6E6E",is_label_show=False,label_color='#6E6E6E',label_text_color ='#6E6E6E',label_formatter='',is_map_symbol_show=True) # type有scatter, effectScatter, heatmap三种模式可选，可根据自己的需求选择对应的图表模式
    
#geo.render(path="中国地级行政区传染病灾害.html")