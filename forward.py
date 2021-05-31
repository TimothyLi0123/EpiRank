""" This module implements the object for a forward run """

import scipy.optimize as opt
from scipy import sparse
import time
import multiprocessing
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from copy import deepcopy
import pandas as pd

from funcs import *

from params import x0
class Forward:
    """ Forward object """

    ncity = 347

    def __init__(self, Fyn, pids=None):
        """ initialise the object """

        self.Fyn = Fyn # type of model

        self.set_params()

        self.init_graph()

        self.model_params()
        self.pids = self.sort_params(pids)
        self.Fscale = 1000.
        self.x0 = x0 ##[0.4, 0.05, 0.2, 0.5, 0.5, 2.0, 1.000, 0.500, 0., 2.000, 2.000, 0.500, 0.100, 0.100, 0.100, 0, 3.000, 1.000, np.array([5]), [8], np.array([[0,1]]), 'l', 2.5, 6, 3]
        self.flow_initialised = False
        self.update_params(self.x0)

        self.checkF()

        self.init_mat()

    def set_params(self):
        """ set universal parameters for the forward model """
        z_city = [0]

        self.params = {}
        self.params['simu_range'] = 0
        self.params['epi'] = [0, 0, 0]
        self.params['zzz'] = [[0], z_city, [[0,0]], 'l']  # 12.19 - 1.1

    def model_params(self):
        """ list of model parameters """

        # parameter id and idx conversion
        ids = ['trc', 'trp'] + ['b'+str(i) for i in range(1,5)]
        tmp = [['F'+str(i)+'cc', 'F'+str(i)+'cp', 'F'+str(i)+'pp'] for i in range(1,5)]
        ids += [item for slist in tmp for item in slist]
        ids += ['z', 'z_city', 'z_length', 'z_pos']
        ids += ['R0', 'DE', 'DI']
        self.para_id = ids
        nparams = len(ids)
        self.pid2idx = {ids[i]: i for i in range(len(ids))}
        self.pidx2id = {i: ids[i] for i in range(len(ids))}

        # indexing of F
        self.Fids = [item for slist in tmp for item in slist]
        self.Fid2idx = {self.Fids[i]:i for i in range(12)}
        self.idx2Fid = {i: self.Fids[i] for i in range(12)}

        # set variables which need to be tracked for computing gradients
        ids = ['dsout', 'deout', 'drout', 'alpha', 'mu', 'eta', 'dein', 'drin', 'dsin', 's', 'e', 'i', 'r']
        self.grad_ids = ids
        # self.gid2idx = {ids[i]: i for i in range(len(ids))}
        # self.gidx2id = {i: ids[i] for i in range(len(ids))}
        self.grad0 = {ids[i]: np.zeros((self.ncity, nparams)) for i in range(len(ids))}
        self.grad1 = {ids[i]: np.zeros((self.ncity, nparams)) for i in range(len(ids))}

    def sort_params(self, ids):
        """ sort parameter ids """

        idx = [self.pid2idx[ids[i]] for i in range(ids.__len__())]
        idx.sort()
        ids = [self.pidx2id[idx[i]] for i in range(idx.__len__())]

        return ids

    def update_params(self, x=None):
        """ update parameters """

        if x is None:
            return

        if self.x0.__len__() == x.__len__():
            self.x0 = x
        else:
            assert self.pids.__len__() == x.__len__(), "number of given parameters is neither all or equal number of specified parameters"
            for k, id in enumerate(self.pids):
                self.x0[self.pid2idx[id]] = x[k]

        self.TR = self.x0[:2]
        self.beta = self.x0[2:6]
        update_flow = len(list(set(self.pids) & set(['F1cc', 'F1cp', 'F1pp', 'F2cc', 'F2cp', 'F2pp', 'F3cc', 'F3cp', 'F3pp', 'F4cc',
 'F4cp', 'F4pp']))) > 0
        if (not self.flow_initialised) or update_flow:
            F_strength = (self.x0[6:9], self.x0[9:12], self.x0[12:15], self.x0[15:18])
            self.init_flow_map(F_strength)
            
        self.params['zzz'] = self.x0[18:22]
        self.params['epi'] = self.x0[22:25]

        self.init_mat()

        for id in self.grad_ids:
            self.grad0[id] *= 0.0
            self.grad1[id] *= 0.0

    def init_graph(self):
        """ initialise the network graph use networkx """

        print('[Graph Preparation].....')

        filename = "data/chinese_cities.csv"
        data = pd.read_csv(filename, encoding="utf-8")

        names_eng = ['Province', 'Name', 'Area', 'Population', 'Population Density', 'GDP', 'GDP year',
                     'GDP per capita', 'Series in Province', 'City level', 'Population Flow 2018', 'Population Flow 2017',
                     'Train 1', 'Train 2', 'Aero flow 50', 'Aero flow 100', 'Lat.', 'Lon.']
        names_chn = list(data.keys()[0:-1])

        self.G = nx.MultiGraph()
        for i in range(self.ncity):
            self.G.add_node(i + 1)
            for j in range(len(names_eng)):
                self.G.node[i + 1][names_eng[j]] = data.iloc[i, j]

        self.city_list_complete = [self.G.node[k]['Name'] for k in self.G.nodes()]
        # city indices
        self.city_list_idx = {self.G.node[k]['Name']: k for k in range(1, self.ncity + 1)}

        self.Province_Capital = list(set([k for k in self.G.nodes() if self.G.node[k]['City level'] in [0, 1, 2]]) | set(
            [k for k in self.G.nodes() if self.G.node[k]['Series in Province'] == 1]))

        # set for central cities and small cities, numbering from 0
        self.Vc = np.zeros(self.ncity)
        for i in self.Province_Capital:
            self.Vc[i-1] = 1
        self.Vp = np.ones(self.ncity) - self.Vc

        self.add_edges()

        # precompute graph information
        """ Difference: now the path list has the same index as the nodes in Graph """
        self.shortest_path_list = cal_shortest_path_list(self.E_all, self.ncity)

        self.cross_infection_list_all = []
        self.dist_q_all = []
        self.check_path_exist = []

        """ modified version of "cal_shortest_path_list", "cross_infection_list" and "dist_q", change to dictionary """
        for q in range(len(self.E_all)):
            cross_infection_list_temp = collections.defaultdict(dict)
            dist_q_temp = collections.defaultdict(dict)
            check_path_exist_temp = collections.defaultdict(dict)

            for j in range(1, self.ncity + 1):
                for i in range(1, self.ncity + 1):
                    cross_infection_list_temp[j - 1][i - 1] = cross_infection_list(j, i, q, self.shortest_path_list, self.ncity)
                    dist_q_temp[j - 1][i - 1] = dist_q(j, i, q, self.shortest_path_list)
                    check_path_exist_temp[j - 1][i - 1] = True if i in self.shortest_path_list[q][j] else False

            self.cross_infection_list_all.append(cross_infection_list_temp)
            self.dist_q_all.append(dist_q_temp)
            self.check_path_exist.append(check_path_exist_temp)

    def add_edges(self):
        """ Add edges to the graph """

        filename = "data/chinese_cities_link_aero.npz"
        data = sparse.load_npz(filename)
        ii, jj = data.nonzero()
        Edges_aero = []
        for k in range(ii.shape[0]):
            self.G.add_edge(ii[k], jj[k], 1, means='aero', color='yellow')
            Edges_aero.append((ii[k], jj[k]))

        print('Aero link ends')

        filename = "data/chinese_cities_link_rail.npz"
        data = sparse.load_npz(filename)
        ii, jj = data.nonzero()
        Edges_rail = []
        for k in range(ii.shape[0]):
            self.G.add_edge(ii[k], jj[k], 2, means='rail', color='black')
            Edges_rail.append((ii[k], jj[k]))

        print('Rail link ends')

        filename = "data/chinese_cities_link_sail.npz"
        data = sparse.load_npz(filename)
        ii, jj = data.nonzero()
        Edges_sail = []
        for k in range(ii.shape[0]):
            self.G.add_edge(ii[k], jj[k], 3, means='sail', color='blue')
            Edges_sail.append((ii[k], jj[k]))

        print('Sail link ends')

        filename = "data/chinese_cities_link_bus.npz"
        data = sparse.load_npz(filename)
        ii, jj = data.nonzero()
        Edges_bus = []
        for k in range(ii.shape[0]):
            self.G.add_edge(ii[k], jj[k], 4, means='bus', color='brown')
            Edges_bus.append((ii[k], jj[k]))

        print('Bus link ends')

        self.E_all = (Edges_aero, Edges_rail, Edges_sail, Edges_bus)

    def init_flow_map(self, F0):
        """ initialise the flow map """

        self.flow_initialised = True
        print('update flows ...')

        F_A_strength, F_R_strength, F_S_strength, F_B_strength = F0

        F_A = np.zeros((self.ncity, self.ncity))
        F_R = np.zeros((self.ncity, self.ncity))
        F_S = np.zeros((self.ncity, self.ncity))
        F_B = np.zeros((self.ncity, self.ncity))
        Fgg = [sparse.lil_matrix((self.ncity, self.ncity)) for i in range(12)]

        for i in range(self.ncity):
            for j in range(self.ncity):

                if i != j:

                    if (j + 1) in self.Province_Capital and (i + 1) in self.Province_Capital:  ##cc

                        if self.dist_q_all[0][j][i] > 0 and self.dist_q_all[0][j][i]!=123456:
                            F_A[i, j] = self.Fscale * F_A_strength[0] / self.dist_q_all[0][j][i]
                            Fgg[0][i, j] = 1. / self.dist_q_all[0][j][i]
                        if self.dist_q_all[1][j][i] > 0 and self.dist_q_all[1][j][i]!=123456:
                            F_R[i, j] = self.Fscale * F_R_strength[0] / self.dist_q_all[1][j][i]
                            Fgg[3][i, j] = 1. / self.dist_q_all[1][j][i]
                        if self.dist_q_all[2][j][i] > 0 and self.dist_q_all[2][j][i]!=123456:
                            F_S[i, j] = self.Fscale * F_S_strength[0] / self.dist_q_all[2][j][i]
                            Fgg[6][i, j] = 1. / self.dist_q_all[2][j][i]
                        if self.dist_q_all[3][j][i] > 0 and self.dist_q_all[3][j][i]!=123456:
                            # F_B[i, j] = self.Fscale * F_B_strength[0] / self.dist_q_all[3][j][i]
                            # Fgg[9][i, j] = 1. / self.dist_q_all[3][j][i]
                            F_B[i, j] = self.Fscale * F_B_strength[0] / bus_distance_factor(j, i)
                            Fgg[9][i, j] = 1. / bus_distance_factor(j, i)

                    elif (j + 1) not in self.Province_Capital and (i + 1) not in self.Province_Capital:  ##pp

                        if self.dist_q_all[0][j][i] > 0 and self.dist_q_all[0][j][i]!=123456:
                            F_A[i, j] = self.Fscale * F_A_strength[2] / self.dist_q_all[0][j][i]
                            Fgg[2][i, j] = 1. / self.dist_q_all[0][j][i]
                        if self.dist_q_all[1][j][i] > 0 and self.dist_q_all[1][j][i]!=123456:
                            F_R[i, j] = self.Fscale * F_R_strength[2] / self.dist_q_all[1][j][i]
                            Fgg[5][i, j] = 1. / self.dist_q_all[1][j][i]
                        if self.dist_q_all[2][j][i] > 0 and self.dist_q_all[2][j][i]!=123456:
                            F_S[i, j] = self.Fscale * F_S_strength[2] / self.dist_q_all[2][j][i]
                            Fgg[8][i, j] = 1. / self.dist_q_all[2][j][i]
                        if self.dist_q_all[3][j][i] > 0 and self.dist_q_all[3][j][i]!=123456:
                            # F_B[i, j] = self.Fscale * F_B_strength[2] / self.dist_q_all[3][j][i]
                            # Fgg[11][i, j] = 1. / self.dist_q_all[3][j][i]
                            F_B[i, j] = self.Fscale * F_B_strength[2] / bus_distance_factor(j, i)
                            Fgg[11][i, j] = 1. / bus_distance_factor(j, i)

                    else:  ##cp

                        if self.dist_q_all[0][j][i] > 0 and self.dist_q_all[0][j][i]!=123456:
                            F_A[i, j] = self.Fscale * F_A_strength[1] / self.dist_q_all[0][j][i]
                            Fgg[1][i, j] = 1. / self.dist_q_all[0][j][i]
                        if self.dist_q_all[1][j][i] > 0 and self.dist_q_all[1][j][i]!=123456:
                            F_R[i, j] = self.Fscale * F_R_strength[1] / self.dist_q_all[1][j][i]
                            Fgg[4][i, j] = 1. / self.dist_q_all[1][j][i]
                        if self.dist_q_all[2][j][i] > 0 and self.dist_q_all[2][j][i]!=123456:
                            F_S[i, j] = self.Fscale * F_S_strength[1] / self.dist_q_all[2][j][i]
                            Fgg[7][i, j] = 1. / self.dist_q_all[2][j][i]
                        if self.dist_q_all[3][j][i] > 0 and self.dist_q_all[3][j][i]!=123456:
                            # F_B[i, j] = self.Fscale * F_B_strength[1] / self.dist_q_all[3][j][i]
                            # Fgg[10][i, j] = 1. / self.dist_q_all[3][j][i]
                            F_B[i, j] = self.Fscale * F_B_strength[1] / bus_distance_factor(j, i)
                            Fgg[10][i, j] = 1. / bus_distance_factor(j, i)

        self.F_raw = [sparse.lil_matrix(F_A), sparse.lil_matrix(F_R), sparse.lil_matrix(F_S), sparse.lil_matrix(F_B)]
        self.F = [F_A, F_R, F_S, F_B]

        # test symmetry
        def is_symmetry(mm):
            err = np.max((mm-mm.T).todense())
            if err > 1.0e-10:
                raise RuntimeError('F not symmetric')

        [is_symmetry(Fgg[i]) for i in range(12)]

        assert np.max(F_A - self.Fscale * (F_A_strength[0]*Fgg[0] + F_A_strength[1]*Fgg[1] + F_A_strength[2]*Fgg[2]).todense()) < 1.0e-10
        assert np.max(
            F_R - self.Fscale * (F_R_strength[0] * Fgg[3] + F_R_strength[1] * Fgg[4] + F_R_strength[2] * Fgg[5]).todense()) < 1.0e-10
        assert np.max(
            F_S - self.Fscale * (F_S_strength[0] * Fgg[6] + F_S_strength[1] * Fgg[7] + F_S_strength[2] * Fgg[8]).todense()) < 1.0e-10
        assert np.max(
            F_B - self.Fscale * (F_B_strength[0] * Fgg[9] + F_B_strength[1] * Fgg[10] + F_B_strength[2] * Fgg[11]).todense()) < 1.0e-10

        self.Fgg = {self.Fids[i]: Fgg[i].todense() for i in range(Fgg.__len__())}
        self.Fgg_out = {self.Fids[i]: np.asarray(np.sum(self.Fgg[self.Fids[i]], axis=0))[0,:] for i in range(Fgg.__len__())}

        self.computeA()

        self.total_flow_out = []
        self.total_flow_in = []
        for i in range(len(self.G)):
            self.total_flow_out.append(
                sum(sum([self.F[q][i, j] for j in range(len(self.G))]) for q in range(len(self.F))))
            self.total_flow_in.append(
                sum(sum([self.F[q][j, i] for j in range(len(self.G))]) for q in range(len(self.F))))

        self.total_flow_out = np.asarray(self.total_flow_out)
        self.total_flow_in = np.asarray(self.total_flow_in)

    def computeA(self):
        """ compute the matrices for big_temp etc... terms """

        self.Ai = [None]*4
        for p in range(4):
            if p in [1, 2]:
                self.Ai[p] = {}
                for i in range(self.ncity):
                    self.Ai[p][i] = sparse.lil_matrix((self.ncity, self.ncity))
                    klist = [k for k in range(self.ncity) if self.check_path_exist[p][k][i]]
                    # compute m1/m3 for all necessary k
                    m13 = {}
                    for k in klist:
                        m1 = self.F[p][k, i]

                        m3 = 0.
                        l_list = self.cross_infection_list_all[p][k][i]
                        for l in l_list:
                            m3 += self.F[p][l, i] * min(self.dist_q_all[p][l][i], self.dist_q_all[p][k][i])

                        if m3 != 0:
                            m13[k] = m1 / m3
                        else:
                            m13[k] = 0.

                    for j in klist:

                        k_list = self.cross_infection_list_all[p][j][i]

                        for k in k_list:
                            m2 = min(self.dist_q_all[p][j][i], self.dist_q_all[p][k][i])

                            self.Ai[p][i][j, k] = m2 * m13[k]
                    self.Ai[p][i] = self.Ai[p][i].tocsc()

        # compute those needed for gradients
        if self.Fyn < 0:
            self.Bi = {}
            self.Ci = {}
            for chi in self.Fids:
                self.Bi[chi] = {}
                self.Ci[chi] = {}
                k = self.Fids.index(chi)
                p = round((k - k % 3) / 3)
                for i in range(self.ncity):
                    self.Bi[chi][i] = sparse.lil_matrix((self.ncity, self.ncity))
                    self.Ci[chi][i] = sparse.lil_matrix((self.ncity, self.ncity))

                    klist = [k for k in range(self.ncity) if self.check_path_exist[p][k][i]]
                    # compute m1/m3 for all necessary k
                    m13, m1g3, m143 = {}, {}, {}
                    for k in klist:
                        m1 = self.F[p][k, i]
                        m1g = self.Fgg[chi][k, i]

                        m3, m4 = 0., 0.
                        l_list = self.cross_infection_list_all[p][k][i]
                        for l in l_list:
                            m3 += self.F[p][l, i] * min(self.dist_q_all[p][l][i], self.dist_q_all[p][k][i])
                            m4 += self.Fgg[chi][l, i] * min(self.dist_q_all[p][l][i], self.dist_q_all[p][k][i])

                        if m3 != 0:
                            m13[k] = m1 / m3
                            m1g3[k] = m1g / m3
                            m143[k] = m1*m4 / m3**2.
                        else:
                            m13[k] = 0.
                            m1g3[k] = 0.
                            m143[k] = 0.

                    for j in klist:

                        k_list = self.cross_infection_list_all[p][j][i]

                        for k in k_list:
                            m2 = min(self.dist_q_all[p][j][i], self.dist_q_all[p][k][i])

                            self.Bi[chi][i][j, k] = m2 * m1g3[k]
                            self.Ci[chi][i][j, k] = m2 * m143[k]
                    self.Bi[chi][i] = self.Bi[chi][i].tocsc()
                    self.Ci[chi][i] = self.Ci[chi][i].tocsc()

    def checkF(self):
        """ check flowmap and connectivity consistency """

        for q in range(len(self.F)):

            F_multi = np.zeros((len(self.G), len(self.G)))
            for k in self.E_all[q]:
                F_multi[k[0] - 1, k[1] - 1] = 1
                F_multi[k[1] - 1, k[0] - 1] = 1

            if q in [1, 2]:  # For rail and sail, check with the transitive closure of F_multi; others check with e2e F_multi
                G_temp = nx.from_numpy_matrix(F_multi)
                DG = G_temp.to_directed()
                DGT = nx.transitive_closure(DG)
                F_multi = np.array(nx.to_numpy_matrix(DGT))

            F_temp = np.multiply(F_multi, self.F[q])
            res = self.F[q] == F_temp
            value, card = np.unique(res, return_counts=True)
            print('Mismatch between Flowmap and connectivity for layer ' + str(q + 1) + ':' + str(card[0]))

            assert np.max(self.F[q]-F_temp) < 1.0e-10
            self.F[q] = F_temp  # update

        self.total_flow_out = []
        self.total_flow_in = []
        for i in range(len(self.G)):
            self.total_flow_out.append(sum(sum([self.F[q][i, j] for j in range(len(self.G))]) for q in range(len(self.F))))
            self.total_flow_in.append(sum(sum([self.F[q][j, i] for j in range(len(self.G))]) for q in range(len(self.F))))
        self.total_flow_out = np.asarray(self.total_flow_out)
        self.total_flow_in = np.asarray(self.total_flow_in)

        print('check F done')

    def init_mat(self):
        """ Initialise matrices for forward runs """

        m, n = self.ncity, self.params['simu_range']+1
        self.ss, self.ee, self.ii, self.rr, self.mu, self.eta = np.zeros((n, m)), np.zeros((n, m)), np.zeros(
            (n, m)), np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m))
        # initial population
        s0 = np.asarray([self.G.node[i + 1]['Population'] * 10000 for i in range(m)])
        self.ss[0, :] = s0
        # self.ss[0, :] = 1.0e+5
        # self.ee[0, :] = 1.0e+5
        # self.ii[0, :] = 1.0e+5
        # self.rr[0, :] = 1.0e+5

        # initialise forcing term
        self.z, self.tildez = np.zeros((n, m)), np.zeros((n, m))
        z_force, z_city, z_length, z_pos = self.params['zzz']
        for i, city in enumerate(z_city):
            d = min(z_length[i][1], self.params['simu_range'])
            self.z[z_length[i][0]:z_length[i][0] + d, z_city[i]] = [z_force[i]] * d
            self.tildez[z_length[i][0]:z_length[i][0] + d, city] = [1.] * d
        
        self.z_pos = z_pos
        
    def simulate(self, x):
        """ One forward run """

        self.update_params(x)
        
        TR = np.ones((self.ncity,)) * self.TR[1]
        for i in range(self.ncity):
            if (i + 1) in self.Province_Capital:
                TR[i] = self.TR[0]

        # print('T = 0/' + str(self.params['simu_range']))

        for t in range(self.params['simu_range']):

            if self.Fyn >= 0:
                # self.one_seir_step(t, TR, self.beta)
                self.one_seir_step_new(t, TR, self.beta, self.z_pos)
                # val = self.one_seir_step_check(t, TR, self.beta)
                # self.one_seir_step_ode(t, TR, self.beta)
            else:
                # t0 = time.time()
                # self.one_seir_step_grad(t, TR, self.beta)
                self.one_seir_step_grad_new(t, TR, self.beta)
                # t1 = time.time()
                # print('total time one step ', t1-t0)
            # print('T = ' + str(t + 1) + '/' + str(self.params['simu_range']))

        output = self.ii + self.ee

        # E + I #
        if self.Fyn >= 0:
            # return val
            return output
        else:
            return output

    def one_seir_step_new(self, t, TR, beta, z_pos):
        """ One time step of the SEIR model """

        R_0, D_E, D_I = self.params['epi']
        mu_temp = self.mu[t, :]
        eta_temp = self.eta[t, :]

        s_temp = self.ss[t, :]
        e_temp = self.ee[t, :]
        i_temp = self.ii[t, :]
        r_temp = self.rr[t, :]
        p_temp = s_temp + e_temp + i_temp + r_temp
        fij = self.total_flow_out

        t0 = time.time()
        big_temp = self.comp_fmu_new(mu_temp, eta_temp, beta)
        t1 = time.time()
        ttemp = t1 - t0

        t0 = time.time()
        
        FF = 0.
        for q in range(self.F.__len__()):
            FF += self.F[q]
        delta_R_in = (1.0 - TR) * FF.dot(eta_temp)
        delta_E_in = (1.0 - TR) * big_temp
        delta_S_in = (1.0 - TR) * fij - delta_E_in - delta_R_in
        
        if z_pos == 'l':  ## zoonotic force in local population
            
            delta_S_out = s_temp / (s_temp + e_temp + r_temp) * (1 - TR) * fij
            delta_E_out = e_temp / (s_temp + e_temp + r_temp) * (1 - TR) * fij
            delta_R_out = r_temp / (s_temp + e_temp + r_temp) * (1 - TR) * fij

            delta_S = -s_temp / p_temp * (R_0 / D_I * i_temp + self.z[t]) + delta_S_in - delta_S_out
            delta_E = s_temp / p_temp * (R_0 / D_I * i_temp + self.z[t]) - e_temp / D_E + delta_E_in - delta_E_out
            delta_I = e_temp / D_E - i_temp / D_I
            delta_R = i_temp / D_I + delta_R_in - delta_R_out
            
        if z_pos == 't':  ## zoonotic force in population in transfer
            
            delta_S_out = s_temp / (s_temp + e_temp + r_temp) * (1 - TR) * fij - self.z[t]
            delta_E_out = e_temp / (s_temp + e_temp + r_temp) * (1 - TR) * fij + self.z[t]
            delta_R_out = r_temp / (s_temp + e_temp + r_temp) * (1 - TR) * fij

            delta_S = -s_temp / p_temp * (R_0 / D_I * i_temp) + delta_S_in - delta_S_out
            delta_E = s_temp / p_temp * (R_0 / D_I * i_temp) - e_temp / D_E + delta_E_in - delta_E_out
            delta_I = e_temp / D_E - i_temp / D_I
            delta_R = i_temp / D_I + delta_R_in - delta_R_out
            
        eta_new = np.zeros(self.ncity)
        mu_new = np.zeros(self.ncity)
        mu_new[fij > 0] = (delta_E_out + TR * big_temp)[fij > 0] / fij[fij > 0]
        eta_new[fij > 0] = (delta_R_out + TR * FF.dot(eta_temp))[fij > 0] / fij[fij > 0]

        self.ss[t + 1, :] = (s_temp + delta_S) * ((s_temp + delta_S) > 0.)
        self.ee[t + 1, :] = (e_temp + delta_E) * ((e_temp + delta_E) > 0.)
        self.ii[t + 1, :] = (i_temp + delta_I) * ((i_temp + delta_I) > 0.)
        self.rr[t + 1, :] = (r_temp + delta_R) * ((r_temp + delta_R) > 0.)
        self.mu[t + 1, :] = mu_new
        self.eta[t + 1, :] = eta_new

        idx = np.asarray(range(self.ncity))
        for i in idx[self.ss[t + 1, :] < 0]:
            raise RuntimeError(
                'Warning! Negative population stock S at city ' + str(i) + ' ' + self.G.node[i + 1]['Name'] + ' at time ' + str(t))
        for i in idx[self.ee[t + 1, :] < 0]:
            raise RuntimeError(
                'Warning! Negative population stock E at city ' + str(i) + ' ' + self.G.node[i + 1]['Name'] + ' at time ' + str(t))
        for i in idx[self.ii[t + 1, :] < 0]:
            raise RuntimeError(
                'Warning! Negative population stock I at city ' + str(i) + ' ' + self.G.node[i + 1]['Name'] + ' at time ' + str(t))
        for i in idx[self.rr[t + 1, :] < 0]:
            raise RuntimeError(
                'Warning! Negative population stock R at city ' + str(i) + ' ' + self.G.node[i + 1]['Name'] + ' at time ' + str(t))

        t1 = time.time()
        trest = t1 - t0

        # print(ttemp, trest)

    def one_seir_step_grad_new(self, t, TR, beta):
        """ One time step of the SEIR model and compute gradients """

        R_0, D_E, D_I = self.params['epi']
        mu_temp = self.mu[t, :]
        eta_temp = self.eta[t, :]

        tt = [0.]*5
        ttemp = np.zeros(4)

        s_temp = self.ss[t, :]
        e_temp = self.ee[t, :]
        i_temp = self.ii[t, :]
        r_temp = self.rr[t, :]
        p_temp = s_temp + e_temp + i_temp + r_temp
        fij = self.total_flow_out

        delta_S_out = s_temp / (s_temp + e_temp + r_temp) * (1 - TR) * fij
        delta_E_out = e_temp / (s_temp + e_temp + r_temp) * (1 - TR) * fij
        delta_R_out = r_temp / (s_temp + e_temp + r_temp) * (1 - TR) * fij

        t0 = time.time()
        self.comp_grad0_new(self.ss[t, :], self.ee[t, :], self.rr[t, :], TR)
        t1 = time.time()
        tt[0] += t1 - t0

        t0 = time.time()
        big_temp, a, ttmp = self.comp_grad1_new(mu_temp, eta_temp, beta)
        ttemp += ttmp
        t1 = time.time()
        tt[1] = t1 - t0

        t0 = time.time()
        self.comp_grad2_new(TR, eta_temp, big_temp, delta_E_out, delta_R_out)
        t1 = time.time()
        tt[2] = t1 - t0

        t0 = time.time()
        self.comp_grad3_new(eta_temp, TR, big_temp)
        t1 = time.time()
        tt[3] += t1 - t0

        FF = 0.
        for q in range(self.F.__len__()):
            FF += self.F[q]
        delta_R_in = (1.0 - TR) * FF.dot(eta_temp)
        delta_E_in = (1.0 - TR) * big_temp
        delta_S_in = (1.0 - TR) * fij - delta_E_in - delta_R_in

        delta_S = -s_temp / p_temp * (R_0 / D_I * i_temp + self.z[t]) + delta_S_in - delta_S_out
        delta_E = s_temp / p_temp * (R_0 / D_I * i_temp + self.z[t]) - e_temp / D_E + delta_E_in - delta_E_out
        delta_I = e_temp / D_E - i_temp / D_I
        delta_R = i_temp / D_I + delta_R_in - delta_R_out

        eta_new = np.zeros(self.ncity)
        mu_new = np.zeros(self.ncity)
        mu_new[fij > 0] = (delta_E_out + TR * big_temp)[fij > 0] / fij[fij > 0]
        eta_new[fij > 0] = (delta_R_out + TR * FF.dot(eta_temp))[fij > 0] / fij[fij > 0]

        t0 = time.time()
        self.comp_grad4_new(s_temp, e_temp, i_temp, r_temp, self.z[t], self.tildez[t], R_0, D_E, D_I, delta_S, delta_E, delta_I, delta_R)
        t1 = time.time()
        tt[4] += t1 - t0

        self.ss[t + 1, :] = (s_temp + delta_S) * ((s_temp + delta_S) > 0.)
        self.ee[t + 1, :] = (e_temp + delta_E) * ((e_temp + delta_E) > 0.)
        self.ii[t + 1, :] = (i_temp + delta_I) * ((i_temp + delta_I) > 0.)
        self.rr[t + 1, :] = (r_temp + delta_R) * ((r_temp + delta_R) > 0.)
        self.mu[t + 1, :] = mu_new
        self.eta[t + 1, :] = eta_new

        idx = np.asarray(range(self.ncity))
        for i in idx[self.ss[t + 1, :] < 0]:
            raise RuntimeError(
                'Warning! Negative population stock S at city ' + str(i) + ' ' + self.G.node[i + 1][
                    'Name'] + ' at time ' + str(t))
        for i in idx[self.ee[t + 1, :] < 0]:
            raise RuntimeError(
                'Warning! Negative population stock E at city ' + str(i) + ' ' + self.G.node[i + 1][
                    'Name'] + ' at time ' + str(t))
        for i in idx[self.ii[t + 1, :] < 0]:
            raise RuntimeError(
                'Warning! Negative population stock I at city ' + str(i) + ' ' + self.G.node[i + 1][
                    'Name'] + ' at time ' + str(t))
        for i in idx[self.rr[t + 1, :] < 0]:
            raise RuntimeError(
                'Warning! Negative population stock R at city ' + str(i) + ' ' + self.G.node[i + 1][
                    'Name'] + ' at time ' + str(t))

        self.grad0 = deepcopy(self.grad1)
        # print(tt)
        # print(ttemp)

    def comp_grad0(self, i, S, E, R, TR):
        """ compute gradients of \Delta S/E/R^out """

        grad = {self.grad_ids[k]: self.grad0[self.grad_ids[k]][i, :] for k in range(len(self.grad_ids))}
        fji, fij = self.total_flow_in[i], self.total_flow_out[i]
        params = set(self.para_id).intersection(self.pids)
        assert fji == fij

        Theta = [S, E, R]
        dTheta = [grad['s'], grad['e'], grad['r']]
        dThetasum = grad['s'] + grad['e'] + grad['r']

        for k, p in enumerate(['dsout', 'deout', 'drout']):
            for chi in params:

                if chi in ['z', 'R0', 'DI', 'DE'] + ['b'+str(i) for i in range(1,5)]:
                    j = self.pid2idx[chi]
                    self.grad1[p][i, j] = (dTheta[k][j]/(S+E+R) - Theta[k]*dThetasum[j]/(S+E+R)**2.) * (1-TR) * fji

                elif chi in ['trc', 'trp']:
                    V = self.Vc if chi == 'trc' else self.Vp
                    j = self.pid2idx[chi]
                    self.grad1[p][i, j] = (dTheta[k][j]/(S+E+R) - Theta[k]*dThetasum[j]/(S+E+R)**2.) * (1-TR) * fji - V[i]*Theta[k]/(S+E+R)*fji

                else:
                    j = self.pid2idx[chi]
                    self.grad1[p][i, j] = (dTheta[k][j] / (S + E + R) - Theta[k] * dThetasum[j] / (S + E + R) ** 2.) * (1 - TR) * fji + self.Fscale * Theta[k]/(S+E+R)*(1-TR)*self.Fgg_out[chi][i]

    def comp_grad0_new(self, S, E, R, TR):
        """ compute gradients of \Delta S/E/R^out """

        grad = deepcopy(self.grad0)
        fji, fij = self.total_flow_in, self.total_flow_out
        params = set(self.para_id).intersection(self.pids)

        Theta = [S, E, R]
        dTheta = [grad['s'], grad['e'], grad['r']]
        dThetasum = grad['s'] + grad['e'] + grad['r']

        for k, p in enumerate(['dsout', 'deout', 'drout']):
            for chi in params:

                if chi in ['z', 'R0', 'DI', 'DE'] + ['b'+str(i) for i in range(1,5)]:
                    j = self.pid2idx[chi]
                    self.grad1[p][:, j] = (dTheta[k][:, j]/(S+E+R) - Theta[k]*dThetasum[:, j]/(S+E+R)**2.) * (1-TR) * fji

                elif chi in ['trc', 'trp']:
                    V = self.Vc if chi == 'trc' else self.Vp
                    j = self.pid2idx[chi]
                    self.grad1[p][:, j] = (dTheta[k][:, j]/(S+E+R) - Theta[k]*dThetasum[:, j]/(S+E+R)**2.) * (1-TR) * fji - V*Theta[k]/(S+E+R)*fji

                else:
                    j = self.pid2idx[chi]
                    self.grad1[p][:, j] = (dTheta[k][:, j] / (S + E + R) - Theta[k] * dThetasum[:, j] / (S + E + R) ** 2.) * (1 - TR) * fji + self.Fscale * Theta[k]/(S+E+R)*(1-TR)*self.Fgg_out[chi]

    def comp_grad1(self, i, mu, eta, beta):
        """ compute gradients for alpha (sum over all q) and big_temp """

        params = set(self.para_id).intersection(self.pids)
        grad = self.grad0

        a = np.zeros((4, len(self.para_id))) # alpha for all q, need to sum them up in the end
        ncity, F, check_path_exist, dist_q_all, cross_infection_list_all = \
            self.ncity, self.F, self.check_path_exist, self.dist_q_all, self.cross_infection_list_all

        tt = np.zeros(4)
        big_temp = 0.
        for p in range(len(F)):
            t0 = time.time()
            if p in [0, 3]:  # air & bus : end to end
                tmp = mu * beta[p] * (1.0 - mu - eta) + mu
                big_temp += F[p][i, :].dot(tmp)

                # compute gradient
                for chi in params:

                    if chi in ['trc', 'trp'] + ['z', 'R0', 'DI', 'DE']:
                        x = self.pid2idx[chi]
                        tmp = grad['mu'][:,x] + beta[p] * ((1.-mu-eta)*grad['mu'][:,x]-mu*(grad['mu'][:,x]+grad['eta'][:,x]))
                        a[p, x] = F[p][i, :].dot(tmp)

                    elif chi in ['b'+str(i) for i in range(1,5)]:
                        x = self.pid2idx[chi]
                        tmp = grad['mu'][:, x] + beta[p] * ((1. - mu - eta) * grad['mu'][:, x] - mu * (grad['mu'][:, x] + grad['eta'][:, x]))
                        bs = ['b'+str(i) for i in range(1,5)]
                        if bs[p] == chi:
                            tmp += mu * (1. - mu - eta)
                        a[p, x] = F[p][i, :].dot(tmp)

                    else:
                        x = self.pid2idx[chi]
                        k = self.Fids.index(chi)
                        tmp = grad['mu'][:, x] + beta[p] * ((1. - mu - eta) * grad['mu'][:, x] - mu * (grad['mu'][:, x] + grad['eta'][:, x]))
                        a[p, x] = F[p][i, :].dot(tmp)
                        if round((k - k % 3)/3) == p:
                            tmp = mu + beta[p] * mu * (1.0 - mu - eta)
                            a[p, x] += self.Fscale * self.Fgg[chi][i, :].dot(tmp)

            else:
                tmp = beta[p] * (1.0 - mu - eta) * (self.Ai[p][i].dot(mu)) + mu
                big_temp += F[p][i, :].dot(tmp)

                # compute gradient
                for chi in params:

                    if chi in ['trc', 'trp'] + ['z', 'R0', 'DI', 'DE']:
                        x = self.pid2idx[chi]
                        tmp = grad['mu'][:, x] + beta[p] * (1.0 - mu - eta) * (self.Ai[p][i].dot(grad['mu'][:, x])) - beta[p]*(grad['mu'][:,x]+grad['eta'][:,x])*(self.Ai[p][i].dot(mu))
                        a[p, x] = F[p][i, :].dot(tmp)

                    elif chi in ['b' + str(i) for i in range(1, 5)]:
                        x = self.pid2idx[chi]
                        tmp = grad['mu'][:, x] + beta[p] * (1.0 - mu - eta) * (self.Ai[p][i].dot(grad['mu'][:, x])) - \
                              beta[p] * (grad['mu'][:, x] + grad['eta'][:, x]) * (self.Ai[p][i].dot(mu))
                        bs = ['b' + str(i) for i in range(1, 5)]
                        if bs[p] == chi:
                            tmp += (1.0 - mu - eta) * (self.Ai[p][i].dot(mu))
                        a[p, x] = F[p][i, :].dot(tmp)

                    else:
                        x = self.pid2idx[chi]
                        k = self.Fids.index(chi)
                        tmp = grad['mu'][:, x] + beta[p] * (1.0 - mu - eta) * (self.Ai[p][i].dot(grad['mu'][:, x])) - \
                              beta[p] * (grad['mu'][:, x] + grad['eta'][:, x]) * (self.Ai[p][i].dot(mu))
                        a[p, x] = F[p][i, :].dot(tmp)
                        if round((k - k % 3) / 3) == p:
                            tmp = mu + beta[p] * (1.0 - mu - eta) * (self.Ai[p][i].dot(mu))
                            a[p, x] += self.Fscale * self.Fgg[chi][i, :].dot(tmp)
                            a[p, x] += self.Fscale * self.F[p][i, :].dot(beta[p] * (1.0 - mu - eta) * (self.Bi[chi][i].dot(mu)))
                            a[p, x] += -self.Fscale * self.F[p][i, :].dot(beta[p] * (1.0 - mu - eta) * (self.Ci[chi][i].dot(mu)))

            t1 = time.time()
            tt[p] = t1-t0

        self.grad1['alpha'][i, :] = np.sum(a, axis=0)

        return big_temp, a, tt

    def comp_grad1_new(self, mu, eta, beta):
        """ compute gradients for alpha (sum over all q) and big_temp """

        params = set(self.para_id).intersection(self.pids)
        grad = deepcopy(self.grad0)

        a = [np.zeros((self.ncity, len(self.para_id))) for i in range(4)] # alpha for all q, need to sum them up in the end
        ncity, F, check_path_exist, dist_q_all, cross_infection_list_all = \
            self.ncity, self.F, self.check_path_exist, self.dist_q_all, self.cross_infection_list_all

        tt = np.zeros(4)
        big_temp = np.zeros(ncity)
        for p in range(len(F)):
            t0 = time.time()
            if p in [0, 3]:  # air & bus : end to end
                tmp = mu * beta[p] * (1.0 - mu - eta) + mu
                big_temp += F[p].dot(tmp)

                # compute gradient
                for chi in params:

                    if chi in ['trc', 'trp'] + ['z', 'R0', 'DI', 'DE']:
                        x = self.pid2idx[chi]
                        tmp = grad['mu'][:,x] + beta[p] * ((1.-mu-eta)*grad['mu'][:,x]-mu*(grad['mu'][:,x]+grad['eta'][:,x]))
                        a[p][:, x] = F[p].dot(tmp)
                        # a[p, x] = F[p][i, :].dot(tmp)

                    elif chi in ['b'+str(i) for i in range(1,5)]:
                        x = self.pid2idx[chi]
                        tmp = grad['mu'][:, x] + beta[p] * ((1. - mu - eta) * grad['mu'][:, x] - mu * (grad['mu'][:, x] + grad['eta'][:, x]))
                        bs = ['b'+str(i) for i in range(1,5)]
                        if bs[p] == chi:
                            tmp += mu * (1. - mu - eta)
                        # a[p, x] = F[p][i, :].dot(tmp)
                        a[p][:, x] = F[p].dot(tmp)

                    else:
                        x = self.pid2idx[chi]
                        k = self.Fids.index(chi)
                        tmp = grad['mu'][:, x] + beta[p] * ((1. - mu - eta) * grad['mu'][:, x] - mu * (grad['mu'][:, x] + grad['eta'][:, x]))
                        # a[p, x] = F[p][i, :].dot(tmp)
                        a[p][:, x] = F[p].dot(tmp)
                        if round((k - k % 3)/3) == p:
                            tmp = mu + beta[p] * mu * (1.0 - mu - eta)
                            # a[p, x] += self.Fgg[chi][i, :].dot(tmp)
                            a[p][:, x] += self.Fscale * np.asarray(self.Fgg[chi].dot(tmp)).reshape(self.ncity)

            else:
                for i in range(ncity):
                    tmp = beta[p] * (1.0 - mu - eta) * (self.Ai[p][i].dot(mu)) + mu
                    big_temp[i] += F[p][i, :].dot(tmp)

                    # compute gradient
                    for chi in params:

                        if chi in ['trc', 'trp'] + ['z', 'R0', 'DI', 'DE']:
                            x = self.pid2idx[chi]
                            tmp = grad['mu'][:, x] + beta[p] * (1.0 - mu - eta) * (self.Ai[p][i].dot(grad['mu'][:, x])) - beta[p]*(grad['mu'][:,x]+grad['eta'][:,x])*(self.Ai[p][i].dot(mu))
                            a[p][i, x] = F[p][i, :].dot(tmp)

                        elif chi in ['b' + str(i) for i in range(1, 5)]:
                            x = self.pid2idx[chi]
                            tmp = grad['mu'][:, x] + beta[p] * (1.0 - mu - eta) * (self.Ai[p][i].dot(grad['mu'][:, x])) - \
                                  beta[p] * (grad['mu'][:, x] + grad['eta'][:, x]) * (self.Ai[p][i].dot(mu))
                            bs = ['b' + str(i) for i in range(1, 5)]
                            if bs[p] == chi:
                                tmp += (1.0 - mu - eta) * (self.Ai[p][i].dot(mu))
                            a[p][i, x] = F[p][i, :].dot(tmp)

                        else:
                            x = self.pid2idx[chi]
                            k = self.Fids.index(chi)
                            tmp = grad['mu'][:, x] + beta[p] * (1.0 - mu - eta) * (self.Ai[p][i].dot(grad['mu'][:, x])) - \
                                  beta[p] * (grad['mu'][:, x] + grad['eta'][:, x]) * (self.Ai[p][i].dot(mu))
                            a[p][i, x] = F[p][i, :].dot(tmp)
                            if round((k - k % 3) / 3) == p:
                                tmp = mu + beta[p] * (1.0 - mu - eta) * (self.Ai[p][i].dot(mu))
                                a[p][i, x] += self.Fscale * self.Fgg[chi][i, :].dot(tmp)
                                a[p][i, x] += self.Fscale * self.F[p][i, :].dot(beta[p] * (1.0 - mu - eta) * (self.Bi[chi][i].dot(mu)))
                                a[p][i, x] += -self.Fscale * self.F[p][i, :].dot(beta[p] * (1.0 - mu - eta) * (self.Ci[chi][i].dot(mu)))

            t1 = time.time()
            tt[p] = t1-t0

        tmp = 0.0
        for p in range(4):
            tmp += a[p]
        self.grad1['alpha'] = tmp

        return big_temp, a, tt

    def comp_grad2(self, i, TR, eta, big_temp, dE, dR):
        """ compute gradients of mu and eta """

        grad = {self.grad_ids[k]: deepcopy(self.grad0[self.grad_ids[k]][i, :]) for k in range(len(self.grad_ids))}
        grad['alpha'] = deepcopy(self.grad1['alpha'][i, :])
        grad['deout'] = deepcopy(self.grad1['deout'][i, :])
        grad['drout'] = deepcopy(self.grad1['drout'][i, :])
        fji, fij = self.total_flow_in[i], self.total_flow_out[i]
        params = set(self.para_id).intersection(self.pids)
        assert fji == fij

        if fji < 1.0e-10:
            self.grad1['mu'][i, :] = 0.
            self.grad1['eta'][i, :] = 0.
            return

        # mu
        for chi in params:

            if chi in ['z', 'R0', 'DI', 'DE'] + ['b' + str(i) for i in range(1, 5)]:
                j = self.pid2idx[chi]
                self.grad1['mu'][i, j] = (grad['deout'][j] + TR * self.grad1['alpha'][i, j]) / fij

            elif chi in ['trc', 'trp']:
                V = self.Vc if chi == 'trc' else self.Vp
                j = self.pid2idx[chi]
                self.grad1['mu'][i, j] = (grad['deout'][j] + TR * self.grad1['alpha'][i, j]) / fij + V[i]*big_temp / fij

            else:
                j = self.pid2idx[chi]
                self.grad1['mu'][i, j] = (grad['deout'][j] + TR * self.grad1['alpha'][i, j]) / fij - self.Fscale * (dE + TR*big_temp) / fij**2. * self.Fgg_out[chi][i]

        # eta
        for chi in params:

            if chi in ['z', 'R0', 'DI', 'DE'] + ['b' + str(i) for i in range(1, 5)]:
                j = self.pid2idx[chi]
                gtmp = 0.
                for q in range(4):
                    gtmp += self.F[q][i, :].dot(self.grad0['eta'][:, j])
                self.grad1['eta'][i, j] = (grad['drout'][j] + TR * gtmp) / fij

            elif chi in ['trc', 'trp']:
                V = self.Vc if chi == 'trc' else self.Vp
                j = self.pid2idx[chi]
                tmp, gtmp = 0., 0.
                for q in range(4):
                    tmp += self.F[q][i, :].dot(eta)
                    gtmp += self.F[q][i, :].dot(self.grad0['eta'][:, j])
                    self.grad1['eta'][i, j] = (grad['drout'][j] + TR * gtmp) / fij + V[i]*tmp / fij

            else:
                j = self.pid2idx[chi]
                tmp, gtmp = 0., 0.
                for q in range(4):
                    tmp += self.F[q][i, :].dot(eta)
                    gtmp += self.F[q][i, :].dot(self.grad0['eta'][:, j])
                self.grad1['eta'][i, j] = (grad['drout'][j] + TR * gtmp + self.Fscale * TR * self.Fgg[chi][i, :].dot(eta)) / fij - self.Fscale * (dR + TR*tmp) / fij**2. * self.Fgg_out[chi][i]

    def comp_grad2_new(self, TR, eta, big_temp, dE, dR):
        """ compute gradients of mu and eta """

        grad = deepcopy(self.grad0)
        grad['alpha'] = deepcopy(self.grad1['alpha'])
        grad['deout'] = deepcopy(self.grad1['deout'])
        grad['drout'] = deepcopy(self.grad1['drout'])
        fji, fij = self.total_flow_in, self.total_flow_out
        params = set(self.para_id).intersection(self.pids)

        eps = 1.0e-14

        # mu
        for chi in params:

            if chi in ['z', 'R0', 'DI', 'DE'] + ['b' + str(i) for i in range(1, 5)]:
                j = self.pid2idx[chi]
                self.grad1['mu'][:, j] = (grad['deout'][:, j] + TR * self.grad1['alpha'][:, j]) / (fij+eps)

            elif chi in ['trc', 'trp']:
                V = self.Vc if chi == 'trc' else self.Vp
                j = self.pid2idx[chi]
                self.grad1['mu'][:, j] = (grad['deout'][:, j] + TR * self.grad1['alpha'][:, j]) / (fij+eps) + V*big_temp / (fij+eps)

            else:
                j = self.pid2idx[chi]
                self.grad1['mu'][:, j] = (grad['deout'][:, j] + TR * self.grad1['alpha'][:, j]) / (fij+eps) - self.Fscale * (dE + TR*big_temp) / (fij+eps)**2. * self.Fgg_out[chi]

        # eta
        for chi in params:

            if chi in ['z', 'R0', 'DI', 'DE'] + ['b' + str(i) for i in range(1, 5)]:
                j = self.pid2idx[chi]
                gtmp = 0.
                for q in range(4):
                    gtmp += self.F[q].dot(self.grad0['eta'][:, j])
                self.grad1['eta'][:, j] = (grad['drout'][:, j] + TR * gtmp) / (fij+eps)

            elif chi in ['trc', 'trp']:
                V = self.Vc if chi == 'trc' else self.Vp
                j = self.pid2idx[chi]
                tmp, gtmp = 0., 0.
                for q in range(4):
                    tmp += self.F[q].dot(eta)
                    gtmp += self.F[q].dot(self.grad0['eta'][:, j])
                    self.grad1['eta'][:, j] = (grad['drout'][:, j] + TR * gtmp) / (fij+eps) + V*tmp / (fij+eps)

            else:
                j = self.pid2idx[chi]
                tmp, gtmp = 0., 0.
                for q in range(4):
                    tmp += self.F[q].dot(eta)
                    gtmp += self.F[q].dot(self.grad0['eta'][:, j])
                self.grad1['eta'][:, j] = (grad['drout'][:, j] + TR * gtmp + self.Fscale * TR * np.asarray(self.Fgg[chi].dot(eta)).reshape(self.ncity)) / (fij+eps) - self.Fscale * (dR + TR*tmp) / (fij+eps)**2. * self.Fgg_out[chi]

            idx = np.arange(self.ncity)
            self.grad1['mu'][idx[fij < eps], :] = 0.
            self.grad1['eta'][idx[fij < eps], :] = 0.

    def comp_grad3(self, i, eta, TR, big_temp):
        """ compute gradients of \delta E/R/S^in """

        grad = {self.grad_ids[k]: self.grad0[self.grad_ids[k]][i, :] for k in range(len(self.grad_ids))}
        grad['alpha'] = deepcopy(self.grad1['alpha'][i, :])
        fji, fij = self.total_flow_in[i], self.total_flow_out[i]
        params = set(self.para_id).intersection(self.pids)
        assert fji == fij

        for chi in params:

            if chi in ['trc', 'trp']:
                V = self.Vc if chi == 'trc' else self.Vp
                j = self.pid2idx[chi]
                self.grad1['dein'][i, j] = (1.-TR)*grad['alpha'][j] - V[i]*big_temp
                tmp, dtmp = 0., 0.
                for q in range(4):
                    tmp += self.F[q][i, :].dot(eta)
                    dtmp += self.F[q][i, :].dot(self.grad0['eta'][:, j])
                self.grad1['drin'][i, j] = (1.-TR)*dtmp - V[i]*tmp
                self.grad1['dsin'][i, j] = -V[i]*fji - self.grad1['dein'][i, j] - self.grad1['drin'][i, j]

            elif chi in ['z', 'R0', 'DI', 'DE'] + ['b' + str(i) for i in range(1, 5)]:
                j = self.pid2idx[chi]
                self.grad1['dein'][i, j] = (1.-TR)*grad['alpha'][j]
                dtmp, tmp = 0., 0.
                for q in range(4):
                    tmp += self.F[q][i, :].dot(eta)
                    dtmp += self.F[q][i, :].dot(self.grad0['eta'][:, j])
                self.grad1['drin'][i, j] = (1.-TR)*dtmp
                self.grad1['dsin'][i, j] = -self.grad1['dein'][i, j] - self.grad1['drin'][i, j]

            else:
                j = self.pid2idx[chi]
                self.grad1['dein'][i, j] = (1. - TR) * grad['alpha'][j]
                dtmp, tmp = 0., 0.
                for q in range(4):
                    tmp += self.F[q][i, :].dot(eta)
                    dtmp += self.F[q][i, :].dot(self.grad0['eta'][:, j])
                self.grad1['drin'][i, j] = (1. - TR) * dtmp + self.Fscale * (1. - TR) * self.Fgg[chi][i, :].dot(eta)
                self.grad1['dsin'][i, j] = self.Fscale * (1. - TR) * self.Fgg_out[chi][i] - self.grad1['dein'][i, j] - self.grad1['drin'][i, j]

    def comp_grad3_new(self, eta, TR, big_temp):
        """ compute gradients of \delta E/R/S^in """

        grad = deepcopy(self.grad0)
        grad['alpha'] = deepcopy(self.grad1['alpha'])
        fji, fij = self.total_flow_in, self.total_flow_out
        params = set(self.para_id).intersection(self.pids)

        for chi in params:

            if chi in ['trc', 'trp']:
                V = self.Vc if chi == 'trc' else self.Vp
                j = self.pid2idx[chi]
                self.grad1['dein'][:, j] = (1.-TR)*grad['alpha'][:, j] - V*big_temp
                tmp, dtmp = 0., 0.
                for q in range(4):
                    tmp += self.F[q].dot(eta)
                    dtmp += self.F[q].dot(self.grad0['eta'][:, j])
                self.grad1['drin'][:, j] = (1.-TR)*dtmp - V*tmp
                self.grad1['dsin'][:, j] = -V*fji - self.grad1['dein'][:, j] - self.grad1['drin'][:, j]

            elif chi in ['z', 'R0', 'DI', 'DE'] + ['b' + str(i) for i in range(1, 5)]:
                j = self.pid2idx[chi]
                self.grad1['dein'][:, j] = (1.-TR)*grad['alpha'][:, j]
                dtmp, tmp = 0., 0.
                for q in range(4):
                    tmp += self.F[q].dot(eta)
                    dtmp += self.F[q].dot(self.grad0['eta'][:, j])
                self.grad1['drin'][:, j] = (1.-TR)*dtmp
                self.grad1['dsin'][:, j] = -self.grad1['dein'][:, j] - self.grad1['drin'][:, j]

            else:
                j = self.pid2idx[chi]
                self.grad1['dein'][:, j] = (1. - TR) * grad['alpha'][:, j]
                dtmp, tmp = 0., 0.
                for q in range(4):
                    tmp += self.F[q].dot(eta)
                    dtmp += self.F[q].dot(self.grad0['eta'][:, j])
                self.grad1['drin'][:, j] = (1. - TR) * dtmp + self.Fscale * (1. - TR) * np.asarray(self.Fgg[chi].dot(eta)).reshape(self.ncity)
                self.grad1['dsin'][:, j] = self.Fscale * (1. - TR) * self.Fgg_out[chi] - self.grad1['dein'][:, j] - self.grad1['drin'][:, j]

    def comp_grad4(self, i, S, E, I, R, z, tildez, R_0, D_E, D_I):
        """ compute gradients of S, E, I, R """

        grad = {self.grad_ids[k]: self.grad0[self.grad_ids[k]][i, :] for k in range(len(self.grad_ids))}
        for p in ['dsout', 'deout', 'drout', 'dsin', 'dein', 'drin']:
            grad[p] = deepcopy(self.grad1[p][i, :])
        params = set(self.para_id).intersection(self.pids)

        P = S + E + I + R
        dP = grad['s'] + grad['e'] + grad['i'] + grad['r']

        for chi in params:
            j = self.pid2idx[chi]
            self.grad1['s'][i, j] = grad['s'][j] - (grad['s'][j]/P - S*dP[j]/P**2.)*(R_0/D_I*I + z) - S/P*R_0/D_I*grad['i'][j] + grad['dsin'][j] - grad['dsout'][j]
            self.grad1['e'][i, j] = grad['e'][j] + (grad['s'][j]/P - S*dP[j]/P**2.)*(R_0/D_I*I + z) + S/P*R_0/D_I*grad['i'][j] - 1./D_E*grad['e'][j] + grad['dein'][j] - grad['deout'][j]
            self.grad1['i'][i, j] = grad['i'][j] + 1./D_E * grad['e'][j] - 1./D_I * grad['i'][j]
            self.grad1['r'][i, j] = grad['r'][j] + 1./D_I * grad['i'][j] + grad['drin'][j] - grad['drout'][j]

            if chi == 'z':
                self.grad1['s'][i, j] += -S/P*tildez
                self.grad1['e'][i, j] += S/P*tildez

            if chi == 'R0':
                self.grad1['s'][i, j] += -S*I/(P*D_I)
                self.grad1['e'][i, j] += S*I/(P*D_I)

            if chi == 'DI':
                self.grad1['s'][i, j] += R_0/D_I**2. * S*I/P
                self.grad1['e'][i, j] += -R_0/D_I**2. * S*I/P
                self.grad1['i'][i, j] += I/D_I**2.
                self.grad1['r'][i, j] += -I/D_I**2.

            if chi == 'DE':
                self.grad1['e'][i, j] += E/D_E**2.
                self.grad1['i'][i, j] += -E/D_E**2.

    def comp_grad4_new(self, S, E, I, R, z, tildez, R_0, D_E, D_I, dS, dE, dI, dR):
        """ compute gradients of S, E, I, R """

        grad = deepcopy(self.grad0)
        for p in ['dsout', 'deout', 'drout', 'dsin', 'dein', 'drin']:
            grad[p] = deepcopy(self.grad1[p])
        params = set(self.para_id).intersection(self.pids)

        P = S + E + I + R
        dP = grad['s'] + grad['e'] + grad['i'] + grad['r']

        for chi in params:
            j = self.pid2idx[chi]
            self.grad1['s'][:, j] = grad['s'][:, j] - (grad['s'][:, j]/P - S*dP[:, j]/P**2.)*(R_0/D_I*I + z) - S/P*R_0/D_I*grad['i'][:, j] + grad['dsin'][:, j] - grad['dsout'][:, j]
            self.grad1['e'][:, j] = grad['e'][:, j] + (grad['s'][:, j]/P - S*dP[:, j]/P**2.)*(R_0/D_I*I + z) + S/P*R_0/D_I*grad['i'][:, j] - 1./D_E*grad['e'][:, j] + grad['dein'][:, j] - grad['deout'][:, j]
            self.grad1['i'][:, j] = grad['i'][:, j] + 1./D_E * grad['e'][:, j] - 1./D_I * grad['i'][:, j]
            self.grad1['r'][:, j] = grad['r'][:, j] + 1./D_I * grad['i'][:, j] + grad['drin'][:, j] - grad['drout'][:, j]

            if chi == 'z':
                self.grad1['s'][:, j] += -S/P*tildez
                self.grad1['e'][:, j] += S/P*tildez

            if chi == 'R0':
                self.grad1['s'][:, j] += -S*I/(P*D_I)
                self.grad1['e'][:, j] += S*I/(P*D_I)

            if chi == 'DI':
                self.grad1['s'][:, j] += R_0/D_I**2. * S*I/P
                self.grad1['e'][:, j] += -R_0/D_I**2. * S*I/P
                self.grad1['i'][:, j] += I/D_I**2.
                self.grad1['r'][:, j] += -I/D_I**2.

            if chi == 'DE':
                self.grad1['e'][:, j] += E/D_E**2.
                self.grad1['i'][:, j] += -E/D_E**2.

            self.grad1['s'][:, j] = self.grad1['s'][:, j] * ((S + dS) > 0.)
            self.grad1['e'][:, j] = self.grad1['e'][:, j] * ((E + dE) > 0.)
            self.grad1['i'][:, j] = self.grad1['i'][:, j] * ((I + dI) > 0.)
            self.grad1['r'][:, j] = self.grad1['r'][:, j] * ((R + dR) > 0.)

    def check_grad(self, chi):
        """ check computation of gradients """

        self.pids = [chi]
        self.Fyn = -1
        self.simulate([self.x0[self.pid2idx[chi]]])
        grad = deepcopy(self.grad1)

        grada = self.exact_grad(chi)

        eps = 1.0e-8
        self.Fyn = 1
        val0 = self.simulate([self.x0[self.pid2idx[chi]]])
        val1 = self.simulate([self.x0[self.pid2idx[chi]]+eps])

        gradc = {self.grad_ids[i]: (val1[self.grad_ids[i]]-val0[self.grad_ids[i]])/eps for i in range(self.grad_ids.__len__())}

        for id in self.grad_ids:
            # if np.linalg.norm(grad[id][:, self.pid2idx[chi]] - gradc[id])/max(np.linalg.norm(gradc[id]), 0.01) > eps:
            #     print(np.linalg.norm(grad[id][:, self.pid2idx[chi]] - gradc[id]))
            #     print('error at ', id)
            # else:
            #     print('ok at ', id, np.linalg.norm(grad[id][:, self.pid2idx[chi]] - gradc[id]))
            print('diff w.r.t. approx ', id, np.max(np.abs(grad[id][:, self.pid2idx[chi]] - gradc[id])))
            # print('diff w.r.t. exact: ', id, np.max(np.abs(grad[id][:, self.pid2idx[chi]] - grada[id])))

    def one_seir_step_check(self, t, TR, beta):
        """ One time step of the SEIR model and compute gradients """


        R_0, D_E, D_I = self.params['epi']
        mu_temp = self.mu[t, :]
        eta_temp = self.eta[t, :]

        val = {self.grad_ids[i]: np.zeros(self.ncity) for i in range(self.grad_ids.__len__())}

        for i in range(self.ncity):

            S_temp = self.ss[t, i]
            E_temp = self.ee[t, i]
            I_temp = self.ii[t, i]
            R_temp = self.rr[t, i]
            P_temp = S_temp + E_temp + I_temp + R_temp

            # big_temp = comp_fmu_c(self.ncity, i, self.F, mu_temp, eta_temp, beta, self.check_path_exist, self.dist_q_all, self.cross_infection_list_all)
            big_temp = self.comp_fmu(i, mu_temp, eta_temp, beta)
            val['alpha'][i] = big_temp

            total_flow = self.total_flow_out[i]

            delta_E_in = (1 - TR[i]) * big_temp


            delta_R_in = 0.
            for q in range(len(self.F)):
                delta_R_in += eta_temp.dot(self.F[q][:, i])
            delta_R_in *= (1 - TR[i])
            delta_S_in = (1 - TR[i]) * self.total_flow_in[i] - delta_R_in - delta_E_in

            val['dein'][i], val['drin'][i], val['dsin'][i] = delta_E_in, delta_R_in, delta_S_in

            total_out = self.total_flow_out[i] - TR[i] * self.total_flow_in[i]

            if total_out < 0:
                print('Warning! Inbound > Outbound at city ' + self.G.node[i + 1]['Name'] + ' at time ' + str(
                    t) + '. Manual zero-out implemented.')
                # break

            delta_E_out = E_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            delta_R_out = R_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            delta_S_out = S_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            val['deout'][i], val['drout'][i], val['dsout'][i] = delta_E_out, delta_R_out, delta_S_out

            #TODO: solve ODE
            delta_S = -S_temp / P_temp * (R_0 / D_I * I_temp + self.z[t][i]) + delta_S_in - delta_S_out
            delta_E = S_temp / P_temp * (R_0 / D_I * I_temp + self.z[t][i]) - E_temp / D_E + delta_E_in - delta_E_out
            delta_I = E_temp / D_E - I_temp / D_I
            delta_R = I_temp / D_I + delta_R_in - delta_R_out

            total_flow = self.total_flow_out[i]

            eps = 1.0e-10
            if total_flow > 0:
                mu_new = (delta_E_out + delta_E_in / max((1 - TR[i]), eps) * TR[i]) / total_flow
                eta_new = (delta_R_out + TR[i] * delta_R_in / max((1 - TR[i]), eps)) / total_flow
            elif total_flow == 0:
                assert (delta_E_out + delta_E_in / max((1 - TR[i]), eps) * TR[i]) == 0
                assert (delta_R_out + TR[i] * delta_R_in / max((1 - TR[i]), eps)) == 0
                mu_new = 0
                eta_new = 0
            else:
                print('Warning: negative total flow out of city ' + str(i) + self.G.node[i + 1]['Name'] + ' at time ' + str(t))

            val['mu'][i], val['eta'][i] = mu_new, eta_new

            S_new = S_temp + delta_S
            E_new = E_temp + delta_E
            I_new = I_temp + delta_I
            R_new = R_temp + delta_R

            val['s'][i], val['e'][i], val['i'][i], val['r'][i] = S_new, E_new, I_new, R_new

            if S_new < 0 or E_new < 0 or I_new < 0 or R_new < 0:
                raise RuntimeError('Warning! Negative population stock at city ' + self.G.node[i + 1]['Name'] + ' at time ' + str(
                    t) + '. Manual zero-out implemented.')

            self.ss[t + 1, i] = S_new
            self.ee[t + 1, i] = E_new
            self.ii[t + 1, i] = I_new
            self.rr[t + 1, i] = R_new
            self.mu[t + 1, i] = mu_new
            self.eta[t + 1, i] = eta_new

        return val

    def exact_grad(self, chi):
        """ exact value of gradient for the built example """

        grad = {self.grad_ids[i]: np.zeros(self.ncity) for i in range(self.grad_ids.__len__())}
        n = self.ncity
        if chi == 'trc':
            if self.params['simu_range'] == 1:
                trc, trp = self.x0[:2]
                tmp = -self.Vc/3.0*self.total_flow_out
                grad['dsout'], grad['deout'], grad['drout'] = deepcopy(tmp), deepcopy(tmp), deepcopy(tmp)
                grad['dsin'] = -self.Vc*self.total_flow_out
                grad['s'] = -2.0/3.0*self.Vc*self.total_flow_out
                grad['e'] = -tmp
                grad['r'] = -tmp
                grad['mu'] = grad['deout']/self.total_flow_out
                grad['eta'] = grad['drout']/self.total_flow_out

            if self.params['simu_range'] == 2:
                trc, trp = self.x0[:2]
                TR = trc*self.Vc + trp*self.Vp
                R_0, D_E, D_I = self.params['epi']
                z = self.z[0, :]
                ser = 1.0e+5*(-1.0/D_E + 1.0/D_I) + 3.0e+5
                f = self.total_flow_out
                grad['dsout'] = (-2.0/3.0*self.Vc*f * (1.0-TR)*f + (1.0e+5-0.25*(R_0/D_I*1.0e+5+z)+2.0/3.0*(1.0-TR)*f)*(-self.Vc*f)) / ser
                grad['deout'] = (1.0/3.0*self.Vc*f * (1.0-TR)*f + (1.0e+5+0.25*(R_0/D_I*1.0e+5+z)-1.0e+5/D_E-1.0/3.0*(1.0-TR)*f)*(-self.Vc*f)) / ser
                grad['drout'] = (1.0/3.0*self.Vc*f * (1.0-TR)*f + (1.0e+5 + 1.0e+5/D_I-1.0/3.0*(1.0-TR)*f)*(-self.Vc*f)) / ser

                tmp, gtmp = 0., 0.
                geta0 = grad['drout']/self.total_flow_out
                eta0 = (1.-TR)
                for q in range(4):
                    gtmp += self.F[q].dot(geta0)
                    tmp += self.F[q].dot(eta0)
                grad['eta'] = (grad['drout'] + self.Vc*tmp +TR*gtmp) / f

        return grad

    def comp_fmu_new(self, mu, eta, beta):
        """ compute gradients for alpha (sum over all q) and big_temp """

        ncity, F, check_path_exist, dist_q_all, cross_infection_list_all = \
            self.ncity, self.F, self.check_path_exist, self.dist_q_all, self.cross_infection_list_all

        big_temp = np.zeros(ncity)
        for p in range(len(F)):
            t0 = time.time()
            if p in [0, 3]:  # air & bus : end to end
                tmp = mu * beta[p] * (1.0 - mu - eta) + mu
                big_temp += F[p].dot(tmp)

            else:
                for i in range(ncity):
                    tmp = beta[p]*(1.0 - mu - eta)*(self.Ai[p][i].dot(mu)) + mu
                    big_temp[i] += F[p][i, :].dot(tmp)

            t1 = time.time()

        return big_temp

    def comp_fmu(self, i, mu, eta, beta):
        """ compute gradients for alpha (sum over all q) and big_temp """

        ncity, F, check_path_exist, dist_q_all, cross_infection_list_all = \
            self.ncity, self.F, self.check_path_exist, self.dist_q_all, self.cross_infection_list_all

        big_temp = 0.
        for p in range(len(F)):
            t0 = time.time()
            if p in [0, 3]:  # air & bus : end to end
                tmp = mu * beta[p] * (1.0 - mu - eta) + mu
                big_temp += F[p][i, :].dot(tmp)


            else:
                tmp = beta[p]*(1.0 - mu - eta)*(self.Ai[p][i].dot(mu)) + mu
                big_temp += F[p][i, :].dot(tmp)

            t1 = time.time()

        return big_temp

    def one_seir_step_grad(self, t, TR, beta):
        """ One time step of the SEIR model and compute gradients """

        R_0, D_E, D_I = self.params['epi']
        mu_temp = self.mu[t, :]
        eta_temp = self.eta[t, :]

        tt = [0.]*5
        ttemp = np.zeros(4)

        for i in range(self.ncity):

            S_temp = self.ss[t, i]
            E_temp = self.ee[t, i]
            I_temp = self.ii[t, i]
            R_temp = self.rr[t, i]
            P_temp = S_temp + E_temp + I_temp + R_temp

            total_out = self.total_flow_out[i] - TR[i] * self.total_flow_in[i]

            if total_out < 0:
                print('Warning! Inbound > Outbound at city ' + self.G.node[i + 1]['Name'] + ' at time ' + str(
                    t) + '. Manual zero-out implemented.')
                # break

            delta_E_out = E_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            delta_R_out = R_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            delta_S_out = S_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            t0 = time.time()
            self.comp_grad0(i, S_temp, E_temp, R_temp, TR[i])
            t1 = time.time()
            tt[0] += t1-t0

            t0 = time.time()
            # big_temp, a = self.comp_grad1(i, mu_temp, eta_temp, beta)
            big_temp, a, ttmp = self.comp_grad1(i, mu_temp, eta_temp, beta)
            ttemp += ttmp
            t1 = time.time()
            tt[1] += t1-t0
            t0 = time.time()
            self.comp_grad2(i, TR[i], eta_temp, big_temp, delta_E_out, delta_R_out)
            t1 = time.time()
            tt[2] += t1-t0
            t0 = time.time()
            self.comp_grad3(i, eta_temp, TR[i], big_temp)
            t1 = time.time()
            tt[3] += t1-t0

            t1 = time.time()

            t0 = time.time()
            total_flow = self.total_flow_out[i]

            delta_E_in = (1 - TR[i]) * big_temp


            delta_R_in = 0.
            for q in range(len(self.F)):
                delta_R_in += eta_temp.dot(self.F[q][:, i])
            delta_R_in *= (1 - TR[i])
            delta_S_in = (1 - TR[i]) * self.total_flow_in[i] - delta_R_in - delta_E_in

            #TODO: solve ODE
            delta_S = -S_temp / P_temp * (R_0 / D_I * I_temp + self.z[t][i]) + delta_S_in - delta_S_out
            delta_E = S_temp / P_temp * (R_0 / D_I * I_temp + self.z[t][i]) - E_temp / D_E + delta_E_in - delta_E_out
            delta_I = E_temp / D_E - I_temp / D_I
            delta_R = I_temp / D_I + delta_R_in - delta_R_out

            total_flow = self.total_flow_out[i]

            eps = 1.0e-10
            if total_flow > 0:
                mu_new = (delta_E_out + delta_E_in / max((1 - TR[i]), eps) * TR[i]) / total_flow
                eta_new = (delta_R_out + TR[i] * delta_R_in / max((1 - TR[i]), eps)) / total_flow
            elif total_flow == 0:
                assert (delta_E_out + delta_E_in / max((1 - TR[i]), eps) * TR[i]) == 0
                assert (delta_R_out + TR[i] * delta_R_in / max((1 - TR[i]), eps)) == 0
                mu_new = 0
                eta_new = 0
            else:
                print('Warning: negative total flow out of city ' + str(i) + self.G.node[i + 1]['Name'] + ' at time ' + str(t))

            S_new = S_temp + delta_S
            E_new = E_temp + delta_E
            I_new = I_temp + delta_I
            R_new = R_temp + delta_R

            t0 = time.time()
            self.comp_grad4(i, S_temp, E_temp, I_temp, R_temp, self.z[t][i], self.tildez[t][i], R_0, D_E, D_I)
            t1 = time.time()
            tt[4] += t1-t0

            if S_new < 0 or E_new < 0 or I_new < 0 or R_new < 0:
                raise RuntimeError('Warning! Negative population stock at city ' + self.G.node[i + 1]['Name'] + ' at time ' + str(
                    t) + '. Manual zero-out implemented.')

            self.ss[t+1, i] = S_new
            self.ee[t+1, i] = E_new
            self.ii[t+1, i] = I_new
            self.rr[t+1, i] = R_new
            self.mu[t+1, i] = mu_new
            self.eta[t+1, i] = eta_new

        self.grad0 = deepcopy(self.grad1)
        # print(tt)
        # print(ttemp)

    def one_seir_step(self, t, TR, beta):
        """ One time step of the SEIR model """

        R_0, D_E, D_I = self.params['epi']
        mu_temp = self.mu[t, :]
        eta_temp = self.eta[t, :]

        ttemp, trest, tsum = 0, 0, 0

        for i in range(self.ncity):

            S_temp = self.ss[t, i]
            E_temp = self.ee[t, i]
            I_temp = self.ii[t, i]
            R_temp = self.rr[t, i]
            P_temp = S_temp + E_temp + I_temp + R_temp

            t0 = time.time()
            # big_temp = comp_fmu(self.ncity, i, self.F, mu_temp, eta_temp, beta, self.check_path_exist, self.dist_q_all, self.cross_infection_list_all)
            # big_temp = comp_fmu_c(self.ncity, i, self.F, mu_temp, eta_temp, beta, self.check_path_exist, self.dist_q_all, self.cross_infection_list_all)
            big_temp = self.comp_fmu(i, mu_temp, eta_temp, beta)

            t1 = time.time()
            ttemp += t1 - t0

            t0 = time.time()
            total_flow = self.total_flow_out[i]

            delta_E_in = (1 - TR[i]) * big_temp


            delta_R_in = 0.
            for q in range(len(self.F)):
                delta_R_in += eta_temp.dot(self.F[q][:, i])
            delta_R_in *= (1 - TR[i])
            delta_S_in = (1 - TR[i]) * self.total_flow_in[i] - delta_R_in - delta_E_in

            total_out = self.total_flow_out[i] - TR[i] * self.total_flow_in[i]

            if total_out < 0:
                print('Warning! Inbound > Outbound at city ' + self.G.node[i + 1]['Name'] + ' at time ' + str(
                    t) + '. Manual zero-out implemented.')
                # break

            delta_E_out = E_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            delta_R_out = R_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            delta_S_out = S_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            #TODO: solve ODE
            delta_S = -S_temp / P_temp * (R_0 / D_I * I_temp + self.z[t][i]) + delta_S_in - delta_S_out
            delta_E = S_temp / P_temp * (R_0 / D_I * I_temp + self.z[t][i]) - E_temp / D_E + delta_E_in - delta_E_out
            delta_I = E_temp / D_E - I_temp / D_I
            delta_R = I_temp / D_I + delta_R_in - delta_R_out

            total_flow = self.total_flow_out[i]

            eps = 1.0e-10
            if total_flow > 0:
                mu_new = (delta_E_out + delta_E_in / max((1 - TR[i]), eps) * TR[i]) / total_flow
                eta_new = (delta_R_out + TR[i] * delta_R_in / max((1 - TR[i]), eps)) / total_flow
            elif total_flow == 0:
                assert (delta_E_out + delta_E_in / max((1 - TR[i]), eps) * TR[i]) == 0
                assert (delta_R_out + TR[i] * delta_R_in / max((1 - TR[i]), eps)) == 0
                mu_new = 0
                eta_new = 0
            else:
                print('Warning: negative total flow out of city ' + str(i) + self.G.node[i + 1]['Name'] + ' at time ' + str(t))

            S_new = S_temp + delta_S
            E_new = E_temp + delta_E
            I_new = I_temp + delta_I
            R_new = R_temp + delta_R

            if S_new < 0 or E_new < 0 or I_new < 0 or R_new < 0:
                raise RuntimeError('Warning! Negative population stock at city ' + self.G.node[i + 1]['Name'] + ' at time ' + str(
                    t) + '. Manual zero-out implemented.')

            self.ss[t+1, i] = S_new
            self.ee[t+1, i] = E_new
            self.ii[t+1, i] = I_new
            self.rr[t+1, i] = R_new
            self.mu[t+1, i] = mu_new
            self.eta[t+1, i] = eta_new

            t1 = time.time()
            trest += t1 - t0

        print(ttemp, trest)

    def one_seir_step_ode(self, t, TR, beta):
        """ One time step of the SEIR model """

        R_0, D_E, D_I = self.params['epi']
        mu_temp = self.mu[t, :]
        eta_temp = self.eta[t, :]

        ttemp, trest, tsum = 0, 0, 0

        for i in range(self.ncity):

            S_temp = self.ss[t, i]
            E_temp = self.ee[t, i]
            I_temp = self.ii[t, i]
            R_temp = self.rr[t, i]
            P_temp = S_temp + E_temp + I_temp + R_temp

            t0 = time.time()
            # big_temp = comp_fmu(self.ncity, i, self.F, mu_temp, eta_temp, beta, self.check_path_exist, self.dist_q_all, self.cross_infection_list_all)
            big_temp = comp_fmu_c(self.ncity, i, self.F, mu_temp, eta_temp, beta, self.check_path_exist, self.dist_q_all, self.cross_infection_list_all)

            t1 = time.time()
            ttemp += t1 - t0

            t0 = time.time()
            total_flow = self.total_flow_out[i]

            delta_E_in = (1 - TR[i]) * big_temp


            delta_R_in = 0.
            for q in range(len(self.F)):
                delta_R_in += eta_temp.dot(self.F[q][:, i])
            delta_R_in *= (1 - TR[i])
            delta_S_in = (1 - TR[i]) * self.total_flow_in[i] - delta_R_in - delta_E_in

            total_out = self.total_flow_out[i] - TR[i] * self.total_flow_in[i]

            if total_out < 0:
                print('Warning! Inbound > Outbound at city ' + self.G.node[i + 1]['Name'] + ' at time ' + str(
                    t) + '. Manual zero-out implemented.')
                # break

            delta_E_out = E_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            delta_R_out = R_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            delta_S_out = S_temp / (S_temp + E_temp + R_temp) * max(0, total_out)

            total_flow = self.total_flow_out[i]

            eps = 1.0e-10
            if total_flow > 0:
                mu_new = (delta_E_out + delta_E_in / max((1 - TR[i]), eps) * TR[i]) / total_flow
                eta_new = (delta_R_out + TR[i] * delta_R_in / max((1 - TR[i]), eps)) / total_flow
            elif total_flow == 0:
                assert (delta_E_out + delta_E_in / max((1 - TR[i]), eps) * TR[i]) == 0
                assert (delta_R_out + TR[i] * delta_R_in / max((1 - TR[i]), eps)) == 0
                mu_new = 0
                eta_new = 0
            else:
                print('Warning: negative total flow out of city ' + str(i) + self.G.node[i + 1]['Name'] + ' at time ' + str(t))

            # # TODO: solve ODE
            # delta_S = -S_temp / P_temp * (R_0 / D_I * I_temp + self.z[t][i]) + delta_S_in - delta_S_out
            # delta_E = S_temp / P_temp * (
            #             R_0 / D_I * I_temp + self.z[t][i]) - E_temp / D_E + delta_E_in - delta_E_out
            # delta_I = E_temp / D_E - I_temp / D_I
            # delta_R = I_temp / D_I + delta_R_in - delta_R_out
            #
            # S_new = S_temp + delta_S
            # E_new = E_temp + delta_E
            # I_new = I_temp + delta_I
            # R_new = R_temp + delta_R

            y0 = [S_temp, E_temp, I_temp, R_temp]
            def f(tau, y):
                S, E, I, R = y
                P = S+E+I+R
                return [-S / P * (R_0 / D_I * I + self.z[t][i]) + delta_S_in - delta_S_out,
                        S / P * (R_0 / D_I * I + self.z[t][i]) - E / D_E + delta_E_in - delta_E_out,
                        E / D_E - I / D_I, I / D_I + delta_R_in - delta_R_out]

            sol = solve_ivp(f, [0, 1], y0, max_step=0.1, t_eval=[1])
            sol = sol.y[:, 0]
            S_new, E_new, I_new, R_new = sol.tolist()

            if S_new < 0 or E_new < 0 or I_new < 0 or R_new < 0:
                raise RuntimeError('Warning! Negative population stock at city ' + self.G.node[i + 1]['Name'] + ' at time ' + str(
                    t) + '. Manual zero-out implemented.')

            self.ss[t+1, i] = S_new
            self.ee[t+1, i] = E_new
            self.ii[t+1, i] = I_new
            self.rr[t+1, i] = R_new
            self.mu[t+1, i] = mu_new
            self.eta[t+1, i] = eta_new

            t1 = time.time()
            trest += t1 - t0

        # print(ttemp, trest)
