# -*- coding : utf-8 -*-
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
import time
from numba import jit
from scipy import sparse

filename = "data/chinese_cities.csv"
cdata = pd.read_csv(filename,encoding="utf-8")

## Misfit ##
def cal_misfit(output, data_inv, z_city, simu_range, city_list_complete):
    # global data_inv,z_city,simu_range

    if len(output[0][1]) < simu_range:  # interrupted by negative stock
        return 123456

    sum_output = sum([output[k][1][-1] for k in range(len(output)) if output[k][0] in [k[0] for k in data_inv]])

    misfit = 0
    for i in range(len([k[0] for k in data_inv])):
        if [kk for kk in range(len(city_list_complete)) if city_list_complete[kk] == [k[0] for k in data_inv][i]][
            0] not in z_city:
            ind = [k[0] for k in output].index([k[0] for k in data_inv][i])

            misfit = misfit + (output[ind][1][-1] / sum_output - data_inv[i][1]) ** 2

    return misfit


## SEIR ##
def Inversion_SEIR(G, shortest_path_list, Para, E_all, F, Province_Capital, cross_infection_list_all, dist_q_all,
                   check_path_exist):
    # global E_all

    # global simu_range

    simu_range = Para[0][1]

    # global R_0,D_E,D_I,z

    R_0 = Para[1][1][0];
    D_E = Para[1][1][1];
    D_I = Para[1][1][2]
    z_force = Para[2][1][0]
    z_city = Para[2][1][1]
    z_length = Para[2][1][2]

    # global R_T

    R_T = Para[3][1]

    # global TR

    TR_strength_Prov = Para[5][1][0]
    TR_strength_Small = Para[5][1][1]

    fake_TR = Para[4][1]

    # global F

    F_A_strength = Para[7][1][0]  # [cc,cp,pp]
    F_R_strength = Para[7][1][1]
    F_S_strength = Para[7][1][2]
    F_B_strength = Para[7][1][3]

    fake_F = Para[6][1]

    z = np.zeros((simu_range, len(G)))

    for i in range(len(z_city)):
        z[0:0 + z_length, z_city[i]] = [z_force] * z_length

    if fake_TR == 1:
        TR = [TR_strength_Small] * len(G)
        for i in range(len(G)):
            if (i + 1) in Province_Capital:
                TR[i] = TR_strength_Prov

    if fake_F == 1:
        F_A = np.zeros((len(G), len(G)))
        F_R = np.zeros((len(G), len(G)))
        F_S = np.zeros((len(G), len(G)))
        F_B = np.zeros((len(G), len(G)))

        for i in range(len(G)):
            for j in range(len(G)):

                if i != j:

                    if (j + 1) in Province_Capital and (i + 1) in Province_Capital:  ##cc

                        if len(shortest_path_list[0][j][i]) > 1:
                            F_A[i, j] = F_A_strength[0] / (len(shortest_path_list[0][j][i]) - 1)
                        if len(shortest_path_list[1][j][i]) > 1:
                            F_R[i, j] = F_R_strength[0] / (len(shortest_path_list[1][j][i]) - 1)
                        if len(shortest_path_list[2][j][i]) > 1:
                            F_S[i, j] = F_S_strength[0] / (len(shortest_path_list[2][j][i]) - 1)
                        if len(shortest_path_list[3][j][i]) > 1:
                            F_B[i, j] = F_B_strength[0] / (len(shortest_path_list[3][j][i]) - 1)

                    elif (j + 1) not in Province_Capital and (i + 1) not in Province_Capital:  ##pp

                        if len(shortest_path_list[0][j][i]) > 1:
                            F_A[i, j] = F_A_strength[2] / (len(shortest_path_list[0][j][i]) - 1)
                        if len(shortest_path_list[1][j][i]) > 1:
                            F_R[i, j] = F_R_strength[2] / (len(shortest_path_list[1][j][i]) - 1)
                        if len(shortest_path_list[2][j][i]) > 1:
                            F_S[i, j] = F_S_strength[2] / (len(shortest_path_list[2][j][i]) - 1)
                        if len(shortest_path_list[3][j][i]) > 1:
                            F_B[i, j] = F_B_strength[2] / (len(shortest_path_list[3][j][i]) - 1)

                    else:  ##cp

                        if len(shortest_path_list[0][j][i]) > 1:
                            F_A[i, j] = F_A_strength[1] / (len(shortest_path_list[0][j][i]) - 1)
                        if len(shortest_path_list[1][j][i]) > 1:
                            F_R[i, j] = F_R_strength[1] / (len(shortest_path_list[1][j][i]) - 1)
                        if len(shortest_path_list[2][j][i]) > 1:
                            F_S[i, j] = F_S_strength[1] / (len(shortest_path_list[2][j][i]) - 1)
                        if len(shortest_path_list[3][j][i]) > 1:
                            F_B[i, j] = F_B_strength[1] / (len(shortest_path_list[3][j][i]) - 1)

    F = [F_A, F_R, F_S, F_B]

    # check flowmap and connectivity consistency #
    for q in range(len(F)):

        F_multi = np.zeros((len(G), len(G)))
        for k in E_all[q]:
            F_multi[k[0] - 1, k[1] - 1] = 1
            F_multi[k[1] - 1, k[0] - 1] = 1

        if q in [1,
                 2]:  # For rail and sail, check with the transitive closure of F_multi; others check with e2e F_multi
            G_temp = nx.from_numpy_matrix(F_multi)
            DG = G_temp.to_directed()
            DGT = nx.transitive_closure(DG)
            F_multi = np.array(nx.to_numpy_matrix(DGT))

        F_temp = np.multiply(F_multi, F[q])
        res = F[q] == F_temp
        value, card = np.unique(res, return_counts=True)
        print('Mismatch between Flowmap and connectivity for layer ' + str(q + 1) + ':' + str(card[0]))

        F[q] = F_temp  # update

    total_flow_out = []
    total_flow_in = []
    for i in range(len(G)):
        total_flow_out.append(sum(sum([F[q][i, j] for j in range(len(G))]) for q in range(len(F))))
        total_flow_in.append(sum(sum([F[q][j, i] for j in range(len(G))]) for q in range(len(F))))

    # initialization #

    for i in range(len(G)):
        G.node[i + 1]['SEIR-S'] = [G.node[i + 1]['Population'] * 10000]
        G.node[i + 1]['SEIR-E'] = [0]
        G.node[i + 1]['SEIR-I'] = [0]
        G.node[i + 1]['SEIR-R'] = [0]
        G.node[i + 1]['SEIR-mu'] = [0]
        G.node[i + 1]['SEIR-eta'] = [0]
        G.node[i + 1]['Transfer Rate'] = TR[i]

    # Simulation #

    # start_s = time.clock()
    print('T = 0/' + str(simu_range))

    for t in range(simu_range):

        G, negative_flag = Simulation_SEIR_TimeStep(G, t + 1, R_0, D_E, D_I, z, R_T, E_all, F, shortest_path_list,
                                                    cross_infection_list_all, dist_q_all, check_path_exist,
                                                    total_flow_out, total_flow_in)
        print('T = ' + str(t + 1) + '/' + str(simu_range))
        if negative_flag == 1:
            break

    # Output #

    output = [[G.node[i + 1]['Name'],
               [G.node[i + 1]['SEIR-I'][m] + G.node[i + 1]['SEIR-E'][m] for m in range(len(G.node[i + 1]['SEIR-I']))]]
              for i in range(len(G))]

    return output


# ## SEIR ##
def Inversion_SEIR_F0(G, shortest_path_list, Para, E_all, F, Province_Capital,
                     cross_infection_list_all, dist_q_all, check_path_exist, total_flow_out, total_flow_in):
    # global simu_range

    # global F

    simu_range = Para[0][1]

    # global R_0,D_E,D_I,z

    R_0 = Para[1][1][0]
    D_E = Para[1][1][1]
    D_I = Para[1][1][2]
    z_force = Para[2][1][0]
    z_city = Para[2][1][1]
    z_length = Para[2][1][2]

    # global R_T

    R_T = Para[3][1]

    # global TR

    TR_strength_Prov = Para[5][1][0]
    TR_strength_Small = Para[5][1][1]

    fake_TR = Para[4][1]

    z = np.zeros((simu_range, len(G)))

    for i in range(len(z_city)):
        z[0:0 + z_length, z_city[i]] = [z_force] * z_length

    if fake_TR == 1:
        TR = [TR_strength_Small] * len(G)
        for i in range(len(G)):
            if (i + 1) in Province_Capital:
                TR[i] = TR_strength_Prov

    # initialization #

    for i in range(len(G)):
        G.node[i + 1]['SEIR-S'] = [G.node[i + 1]['Population'] * 10000]
        G.node[i + 1]['SEIR-E'] = [0]
        G.node[i + 1]['SEIR-I'] = [0]
        G.node[i + 1]['SEIR-R'] = [0]
        G.node[i + 1]['SEIR-mu'] = [0]
        G.node[i + 1]['SEIR-eta'] = [0]
        G.node[i + 1]['Transfer Rate'] = TR[i]

    # Simulation #

    start_s = time.clock()
    print('T = 0/' + str(simu_range))
    ss, ee, ii, rr, mu, eta = np.zeros((simu_range+1, 347)), np.zeros((simu_range+1, 347)), np.zeros((simu_range+1, 347)), np.zeros((simu_range+1, 347)), np.zeros((simu_range+1, 347)), np.zeros((simu_range+1, 347))
    ss[0, :] = np.asarray([G.node[i + 1]['SEIR-S'][0] for i in range(347)])

    for t in range(simu_range):
    # for t in [0, 1, 2]:

        G, negative_flag = Simulation_SEIR_TimeStep(G, t + 1, R_0, D_E, D_I, z, R_T, E_all, F, shortest_path_list,
                                                    cross_infection_list_all, dist_q_all, check_path_exist,
                                                    total_flow_out, total_flow_in)
        ss[t+1, :] = np.asarray([G.node[i + 1]['SEIR-S'][t+1] for i in range(347)])
        ee[t + 1, :] = np.asarray([G.node[i + 1]['SEIR-E'][t + 1] for i in range(347)])
        ii[t + 1, :] = np.asarray([G.node[i + 1]['SEIR-I'][t + 1] for i in range(347)])
        rr[t + 1, :] = np.asarray([G.node[i + 1]['SEIR-R'][t + 1] for i in range(347)])
        mu[t + 1, :] = np.asarray([G.node[i + 1]['SEIR-mu'][t + 1] for i in range(347)])
        eta[t + 1, :] = np.asarray([G.node[i + 1]['SEIR-eta'][t + 1] for i in range(347)])

        print('T = ' + str(t + 1) + '/' + str(simu_range))

        if negative_flag == 1:
            break

    # Output #
    np.save('ss', ss)
    np.save('ee', ee)
    np.save('ii', ii)
    np.save('rr', rr)
    np.save('mu', mu)
    np.save('eta', eta)

    output = [[G.node[i + 1]['Name'],
               [G.node[i + 1]['SEIR-I'][m] + G.node[i + 1]['SEIR-E'][m] for m in range(len(G.node[i + 1]['SEIR-I']))]]
              for i in range(len(G))]
    # E + I #

    return output


## SEIR ##
def Inversion_SEIR_F(G, shortest_path_list, Para, E_all, F, Province_Capital,
                     cross_infection_list_all, dist_q_all, check_path_exist, total_flow_out, total_flow_in):
    # global simu_range

    # global F

    simu_range = Para[0][1]

    # global R_0,D_E,D_I,z

    # R_0 = Para[1][1][0]
    # D_E = Para[1][1][1]
    # D_I = Para[1][1][2]
    R_0, D_E, D_I = Para['rdd']
    # z_force = Para[2][1][0]
    # z_city = Para[2][1][1]
    # z_length = Para[2][1][2]
    z_force, z_city, z_length = Para['zzz']

    # global R_T

    # R_T = Para[3][1]
    R_T = Para['R_T']

    # global TR

    # TR_strength_Prov = Para[5][1][0]
    # TR_strength_Small = Para[5][1][1]
    TR_strength_Prov = Para['TR'][0]
    TR_strength_Small = Para['TR'][1]

    # fake_TR = Para[4][1]
    fake_TR = Para['TR_handle']

    z = np.zeros((simu_range, len(G)))

    for i in range(len(z_city)):
        z[0:0 + z_length, z_city[i]] = [z_force] * z_length

    # if fake_TR == 1:
    #     TR = [TR_strength_Small] * len(G)
    #     for i in range(len(G)):
    #         if (i + 1) in Province_Capital:
    #             TR[i] = TR_strength_Prov

    if fake_TR == 1:
        TR = np.ones((len(G),)) * TR_strength_Small
        for i in range(len(G)):
            if (i + 1) in Province_Capital:
                TR[i] = TR_strength_Prov

    # initialization #

    # for i in range(len(G)):
    #     G.node[i + 1]['SEIR-S'] = [G.node[i + 1]['Population'] * 10000]
    #     G.node[i + 1]['SEIR-E'] = [0]
    #     G.node[i + 1]['SEIR-I'] = [0]
    #     G.node[i + 1]['SEIR-R'] = [0]
    #     G.node[i + 1]['SEIR-mu'] = [0]
    #     G.node[i + 1]['SEIR-eta'] = [0]
    #     G.node[i + 1]['Transfer Rate'] = TR[i]

    # initial condition of seir_s
    seir_s_init = np.asarray([G.node[i + 1]['Population'] * 10000 for i in range(len(G))])

    # Simulation #

    # start_s = time.clock()
    print('T = 0/' + str(simu_range))

    for t in range(simu_range):

        G, negative_flag = Simulation_SEIR_TimeStep(G, t + 1, R_0, D_E, D_I, z, R_T, E_all, F, shortest_path_list,
                                                    cross_infection_list_all, dist_q_all, check_path_exist,
                                                    total_flow_out, total_flow_in)
        print('T = ' + str(t + 1) + '/' + str(simu_range))

        if negative_flag == 1:
            # break
            raise RuntimeError('Negative population stock encountered')

    # Output #

    output = [[G.node[i + 1]['Name'],
               [G.node[i + 1]['SEIR-I'][m] + G.node[i + 1]['SEIR-E'][m] for m in range(len(G.node[i + 1]['SEIR-I']))]]
              for i in range(len(G))]
    # E + I #

    return output


# def cal_shortest_path_list0(E_all,len_city_list):
#
#     #global len_city_list
#
#     shortest_path_list = []
#     for q in range(len(E_all)):
#
#         ## Single Layer ##
#         G_single = nx.Graph()
#         G_single.add_nodes_from([k+1 for k in list(range(len_city_list))])
#         G_single.add_edges_from([e[0:2] for e in E_all[q]])
#
#         shortest_path_temp = [[[123456] for col in range(len_city_list)] for row in range(len_city_list)]
#         for i in range(len_city_list):
#             for j in range(len_city_list):
#                 if nx.has_path(G_single,j+1,i+1):
#                     if q in [0,3]:   # air & bus : end to end
#                         shortest_path_temp[j][i] = [j+1,i+1]
#                     else:
#                         shortest_path_temp[j][i] = nx.shortest_path(G_single,j+1,i+1)
#         shortest_path_list.append(shortest_path_temp)
#
#     return shortest_path_list


def cal_shortest_path_list(E_all, len_city_list):
    """ Modified version: Compute the shortest path between cities for each transportation method
        Speed up, output form: shortest_path_list[q][i][j]
        Achtung: now the path list has the same index as the nodes in Graph, no need to +1 while using """
    # global len_city_list

    shortest_path_list = []
    for q in range(len(E_all)):

        # air & bus, only direct connection is considered
        if q in [0, 3]:
            shortest_path_q = collections.defaultdict(dict)
            for i in range(len_city_list):
                shortest_path_q[i + 1][i + 1] = [i + 1]  # to be consistent with output of nx.shortest_path(G)
            for e in E_all[q]:
                shortest_path_q[e[0]][e[1]] = [e[0], e[1]]
                shortest_path_q[e[1]][e[0]] = [e[1], e[0]]

        else:
            ## Single Layer ##
            G_single = nx.Graph()
            G_single.add_nodes_from([k + 1 for k in list(range(len_city_list))])
            G_single.add_edges_from([e[0:2] for e in E_all[q]])

            shortest_path_q = nx.shortest_path(G_single)
            for i in range(len_city_list):
                for j in range(len_city_list):
                    if i+1 in shortest_path_q[j+1]:
                        # TODO: settle issue with multiple shortest path
                        tmp = nx.shortest_path(G_single,j+1,i+1)
                        shortest_path_q[j+1][i+1] = tmp
                        # shortest_path_q[j+1][i+1] = tmp[::-1]

        shortest_path_list.append(shortest_path_q)

    return shortest_path_list


# def cross_infection_list0(j,i,q,shortest_path_list,len_city_list):
#
#     #global shortest_path_list,len_city_list
#
#     if q in [0,3]:   # air & bus : end to end
#         return [j]
#
#     k_list = []
#
#     j_path = shortest_path_list[q][j][i]
#
#     for k in range(len_city_list):
#         if shortest_path_list[q][k][i] != [123456]:
#             if (j+1) in shortest_path_list[q][k][i] or (k+1) in j_path:
#                 k_list.append(k)
#
#     return k_list


def cross_infection_list(j, i, q, shortest_path_list, len_city_list):
    """ compute cross infection list from j to i with transportation method q """
    # global shortest_path_list,len_city_list

    if q in [0, 3]:
        if i in shortest_path_list[q][j]:
            return [j-1]
        else:
            return []

    k_list = []

    if i in shortest_path_list[q][j]:
        j_path = shortest_path_list[q][j][i]

        for k in range(1, len_city_list + 1):
            if i in shortest_path_list[q][k]:
                if (j in shortest_path_list[q][k][i] or k in j_path) and k!=i:
                    k_list.append(k-1)

    return k_list


# def dist_q0(j,i,q,shortest_path_list):
#
#     #global shortest_path_list
#
#     if shortest_path_list[q][j][i] == [123456]:
#         return 123456
#     else:
#         return len(shortest_path_list[q][j][i]) - 1


def dist_q(j, i, q, shortest_path_list):
    # global shortest_path_list

    if i not in shortest_path_list[q][j]:
        return 123456
    else:
        return len(shortest_path_list[q][j][i]) - 1


def Simulation_SEIR_TimeStep(G, t, R_0, D_E, D_I, z, R_T, E_all, F, shortest_path_list,
                             cross_infection_list_all, dist_q_all, check_path_exist, total_flow_out, total_flow_in):
    negative_flag = 0  # flag of negative stock

    for i in range(len(G)):

        S_temp = G.node[i + 1]['SEIR-S'][t - 1]
        E_temp = G.node[i + 1]['SEIR-E'][t - 1]
        I_temp = G.node[i + 1]['SEIR-I'][t - 1]
        R_temp = G.node[i + 1]['SEIR-R'][t - 1]
        P_temp = S_temp + E_temp + I_temp + R_temp

        mu_temp = G.node[i + 1]['SEIR-mu'][t - 1]
        eta_temp = G.node[i + 1]['SEIR-eta'][t - 1]

        big_temp = 0

        for q in range(len(F)):

            for j in range(len(G)):

                if check_path_exist[q][j][i] == False:
                    continue

                if q in [0, 3]:  # air & bus : end to end
                    big_temp = big_temp + F[q][j, i] * G.node[j + 1]['SEIR-mu'][t - 1] * R_T[q]
                    continue

                first_term = F[q][j, i] * G.node[j + 1]['SEIR-mu'][t - 1]

                second_term = (R_T[q] - 1)

                sum_second_term = 0

                k_list = cross_infection_list_all[q][j][i]

                for k in k_list:

                    l_list = cross_infection_list_all[q][k][i]

                    m1 = F[q][k, i] * G.node[k + 1]['SEIR-mu'][t - 1]
                    m2 = F[q][j, i] * (1 - G.node[j + 1]['SEIR-mu'][t - 1] - G.node[j + 1]['SEIR-eta'][t - 1]) * min(
                        dist_q_all[q][j][i], dist_q_all[q][k][i])
                    m3 = sum([F[q][l, i] * (
                                1 - G.node[l + 1]['SEIR-mu'][t - 1] - G.node[l + 1]['SEIR-eta'][t - 1]) * min(
                        dist_q_all[q][l][i], dist_q_all[q][k][i]) for l in l_list])

                    if m3 != 0:
                        sum_second_term = sum_second_term + m1 * m2 / m3
                    else:
                        sum_second_term = sum_second_term

                big_temp = big_temp + (first_term + second_term * sum_second_term)

        total_flow = total_flow_out[i]

        total_out = (total_flow_out[i] - G.node[i + 1]['Transfer Rate'] * total_flow_in[i])

        delta_E_in = (1 - G.node[i + 1]['Transfer Rate']) * big_temp

        delta_R_in = (1 - G.node[i + 1]['Transfer Rate']) * sum(
            [sum([F[q][j, i] * G.node[j + 1]['SEIR-eta'][t - 1] for j in range(len(G))]) for q in range(len(F))])

        delta_S_in = (1 - G.node[i + 1]['Transfer Rate']) * total_flow_in[i] - delta_R_in - delta_E_in

        total_out = (sum(sum([F[q][i, j] for j in range(len(G))]) for q in range(len(F))) \
                     - G.node[i + 1]['Transfer Rate'] * sum(
                    [sum([F[q][j, i] for j in range(len(G))]) for q in range(len(F))]))

        if total_out < 0:
            print('Warning! Inbound > Outbound at city ' + G.node[i + 1]['Name'] + ' at time ' + str(
                t) + '. Manual zero-out implemented.')
            negative_flag = 1
            break

        delta_E_out = E_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

        delta_R_out = R_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

        delta_S_out = S_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

        delta_S = -S_temp / P_temp * (R_0 / D_I * I_temp + z[t - 1][i]) + delta_S_in - delta_S_out
        delta_E = S_temp / P_temp * (R_0 / D_I * I_temp + z[t - 1][i]) - E_temp / D_E + delta_E_in - delta_E_out
        delta_I = E_temp / D_E - I_temp / D_I
        delta_R = I_temp / D_I + delta_R_in - delta_R_out

        total_flow = (sum(sum([F[q][i, j] for j in range(len(G))]) for q in range(len(F))))

        if total_flow > 0:
            mu_new = (delta_E_out + delta_E_in / (1 - G.node[i + 1]['Transfer Rate']) * G.node[i + 1][
                'Transfer Rate']) / total_flow
            eta_new = (delta_R_out + G.node[i + 1]['Transfer Rate'] * sum(
                [sum([F[q][j, i] * G.node[j + 1]['SEIR-eta'][t - 1] for j in range(len(G))]) for q in
                 range(len(F))])) / total_flow
        else:
            mu_new = 0
            eta_new = 0

        S_new = S_temp + delta_S
        E_new = E_temp + delta_E
        I_new = I_temp + delta_I
        R_new = R_temp + delta_R

        if S_new < 0 or E_new < 0 or I_new < 0 or R_new < 0:
            print('Warning! Negative population stock at city ' + G.node[i + 1]['Name'] + ' at time ' + str(
                t) + '. Manual zero-out implemented.')
            negative_flag = 1
            break

        G.node[i + 1]['SEIR-S'].append(max(S_new, 0))
        G.node[i + 1]['SEIR-E'].append(max(E_new, 0))
        G.node[i + 1]['SEIR-I'].append(max(I_new, 0))
        G.node[i + 1]['SEIR-R'].append(max(R_new, 0))
        G.node[i + 1]['SEIR-mu'].append(mu_new)
        G.node[i + 1]['SEIR-eta'].append(eta_new)

    return G, negative_flag


from math import sin, asin, cos, radians, fabs, sqrt


def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance_km(lat0, lng0, lat1, lng1):
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * 6378.137 * asin(sqrt(h))

    return distance


def comp_fmu(ncity, i, F, mu_temp, eta_temp, beta, check_path_exist, dist_q_all, cross_infection_list_all):
    """ compute the 'big_temp' term """

    big_temp = 0.
    t1, t2 = 0, 0
    for q in range(len(F)):
        if q in [0, 3]: # air & bus : end to end
            for j in range(ncity):
                if check_path_exist[q][j][i] == False:
                    continue
                # big_temp = big_temp + self.F[q][j, i] * G.node[j + 1]['SEIR-mu'][t - 1] * R_T[q]
                big_temp += F[q][j, i] * mu_temp[j] * (beta[q]+1)

        else:
            klist = [k for k in range(ncity) if check_path_exist[q][k][i]]
            # compute m1/m3 for all necessary k
            m13 = {}
            for k in klist:
                m1 = F[q][k, i] * mu_temp[k]

                m3 = 0.
                l_list = cross_infection_list_all[q][k][i]
                for l in l_list:
                    m3 += F[q][l, i] * (1 - mu_temp[l] - eta_temp[l]) * min(dist_q_all[q][l][i], dist_q_all[q][k][i])

                if m3 != 0:
                    m13[k] = m1 / m3
                else:
                    m13[k] = 0.

            for j in klist:

                # first_term = F[q][j, i] * G.node[j + 1]['SEIR-mu'][t - 1]
                first_term = F[q][j, i] * mu_temp[j]

                second_term = beta[q]

                sum_second_term = 0.

                k_list = cross_infection_list_all[q][j][i]

                for k in k_list:

                    m2 = F[q][j, i] * (1 - mu_temp[j] - eta_temp[j]) * min(dist_q_all[q][j][i], dist_q_all[q][k][i])

                    sum_second_term += m2 * m13[k]

                big_temp = big_temp + (first_term + second_term * sum_second_term)

    return big_temp


def comp_fmu_c(ncity, i, F, mu_temp, eta_temp, beta, check_path_exist, dist_q_all, cross_infection_list_all):
    """ compute the 'big_temp' term """

    big_temp = 0.
    t1, t2 = 0, 0
    for q in range(len(F)):
        if q in [0, 3]: # air & bus : end to end
            for j in range(ncity):
                if check_path_exist[q][j][i] == False:
                    continue
                # big_temp += F[q][j, i] * mu_temp[j] * R_T[q]
                big_temp += F[q][j, i] * mu_temp[j] * beta[q] * (1.0 - mu_temp[j] - eta_temp[j]) + F[q][j, i] * mu_temp[j]

        else:
            klist = [k for k in range(ncity) if check_path_exist[q][k][i]]
            # compute m1/m3 for all necessary k
            m13 = {}
            for k in klist:
                m1 = F[q][k, i] * mu_temp[k]

                m3 = 0.
                l_list = cross_infection_list_all[q][k][i]
                for l in l_list:
                    # m3 += F[q][l, i] * (1 - mu_temp[l] - eta_temp[l]) * min(dist_q_all[q][l][i], dist_q_all[q][k][i])
                    m3 += F[q][l, i] * min(dist_q_all[q][l][i], dist_q_all[q][k][i])

                if m3 != 0:
                    m13[k] = m1 / m3
                else:
                    m13[k] = 0.

            for j in klist:

                # first_term = F[q][j, i] * G.node[j + 1]['SEIR-mu'][t - 1]
                first_term = F[q][j, i] * mu_temp[j]

                second_term = beta[q]

                sum_second_term = 0.

                k_list = cross_infection_list_all[q][j][i]

                for k in k_list:

                    m2 = F[q][j, i] * (1 - mu_temp[j] - eta_temp[j]) * min(dist_q_all[q][j][i], dist_q_all[q][k][i])

                    sum_second_term += m2 * m13[k]

                big_temp = big_temp + (first_term + second_term * sum_second_term)

    return big_temp


def bus_distance_factor(i, j):
    """ the normalisation factor for bus flow strength """

    lat1 = cdata.iloc[i, -3]
    lon1 = cdata.iloc[i, -2]
    lat2 = cdata.iloc[j, -3]
    lon2 = cdata.iloc[j, -2]

    dis = get_distance_km(lat1, lon1, lat2, lon2)

    # return max(dis/50., 1.)
    # return max(np.exp(dis / 50. - 1), 1.)
    return 1.

