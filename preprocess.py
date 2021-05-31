""" This module preprocesses data for WH-inversion
    For consistency with other parts of the code, counting starts from 1 """

from scipy import sparse
import numpy as np
import pandas as pd

from funcs import *

filename = "data/chinese_cities.csv"
cdata = pd.read_csv(filename,encoding="utf-8")

names_eng=['Province','Name','Area','Population','Population Density','GDP','GDP year','GDP per capita','Series in Province','City level',\
       'Population Flow 2018','Population Flow 2017','Train 1','Train 2','Aero flow 50','Aero flow 100','Lat.','Lon.']
names_chn = list(cdata.keys()[0:-1])

city_list_complete = list(cdata.iloc[:, 1])[:-1]
len_city_list = city_list_complete.__len__()
citytag2idx = {city_list_complete[k]: k+1 for k in range(len_city_list)}
idx2citytag = {k+1: city_list_complete[k] for k in range(len_city_list)}

Province_Capital = list(set([k for k in range(1,len_city_list+1) if cdata.iloc[k-1,9] in [0,1,2]]) | set([k for k in range(1,len_city_list+1) if cdata.iloc[k-1,8] == 1]))


def zerodiag(mat):
    """ set diagonal of mat to be zero """

    for i in range(mat.shape[0]):
        mat[i, i] = 0

    return mat


def aero_link():
    """ Preprocess aero links """

    filename = "data/chinese_cities_link_aero.csv"
    data = pd.read_csv(filename, encoding="utf-8")

    departure_list = list(data.keys())
    departure_idx = [i for i in range(len(departure_list)) if departure_list[i] in city_list_complete]

    mat = sparse.lil_matrix((len_city_list+1, len_city_list+1))
    for i in departure_idx:
        destination_list = list(data.iloc[:, i])
        destination_list = list(set(destination_list).intersection(set(city_list_complete)))
        for j in range(len(destination_list)):
            mat[citytag2idx[departure_list[i]], citytag2idx[destination_list[j]]] = 1

    mat = zerodiag(mat)
    sparse.save_npz('data/chinese_cities_link_aero.npz', sparse.csr_matrix(mat))


def rail_link():
    """ Preprocess rail links """

    filename = "data/chinese_cities_link_rail.csv"
    data = pd.read_csv(filename, encoding="utf-8")

    rail_list = list(data.keys())

    mat = sparse.lil_matrix((len_city_list+1, len_city_list+1))
    for i in range(len(rail_list)):
        rail_station = list(data.iloc[:,i])
        rail_station = [x for x in rail_station if str(x) != 'nan']
        idx = [citytag2idx[rail_station[k]] for k in range(len(rail_station))]
        edges = [(idx[jj], idx[jj+1]) for jj in range(len(idx)-1)]

        for e in edges:
            mat[e[0], e[1]] = 1

    mat = zerodiag(mat)
    sparse.save_npz('data/chinese_cities_link_rail.npz', sparse.csr_matrix(mat))


def sail_link():
    """ Preprocess sail links """

    filename = "data/chinese_cities_link_sail.csv"
    data = pd.read_csv(filename, encoding="utf-8")

    sail_list = list(data.keys())

    mat = sparse.lil_matrix((len_city_list + 1, len_city_list + 1))
    for i in range(len(sail_list)):
        sail_port = list(data.iloc[:, i])
        sail_port = [x for x in sail_port if str(x) != 'nan']
        idx = [citytag2idx[sail_port[k]] for k in range(len(sail_port))]
        edges = [(idx[jj], idx[jj+1]) for jj in range(len(idx)-1)]

        for e in edges:
            mat[e[0], e[1]] = 1

    mat = zerodiag(mat)
    sparse.save_npz('data/chinese_cities_link_sail.npz', sparse.csr_matrix(mat))


def bus_link():
    """ Preprocess bus links """

    dis_threshold = 150
    # dis_threshold = 80

    mat = sparse.lil_matrix((len_city_list + 1, len_city_list + 1))
    for i in range(1, len_city_list+1):
        for j in range(i, len_city_list+1):

            # latitude and longitude
            lat1 = cdata.iloc[i-1, -3]
            lon1 = cdata.iloc[i-1, -2]
            lat2 = cdata.iloc[j-1, -3]
            lon2 = cdata.iloc[j-1, -2]

            dis = get_distance_km(lat1, lon1, lat2, lon2)

            if dis < dis_threshold and dis > 0:
            # if dis < dis_threshold and dis > 0 and cdata.iloc[i-1, 0] == cdata.iloc[j-1, 0]:
            #     if i == 9:
                    # print(cdata.iloc[j-1,1])
                mat[i, j] = 1

            # if cdata.iloc[i-1, 0] == cdata.iloc[j-1, 0] and (cdata.iloc[i-1, 8] == 1 or cdata.iloc[j-1, 8] == 1) and dis < 2*dis_threshold:
            if cdata.iloc[i-1, 0] == cdata.iloc[j-1, 0] and (cdata.iloc[i-1, 8] == 1 or cdata.iloc[j-1, 8] == 1):
            #     if i == 9:
                    # print(cdata.iloc[j - 1, 1])
                mat[i, j] = 1

    mat = zerodiag(mat)
    sparse.save_npz('data/chinese_cities_link_bus.npz', sparse.csr_matrix(mat))


aero_link()
rail_link()
sail_link()
bus_link()
print('done')
