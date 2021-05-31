""" Example usage of the forward run """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

""" Preprocessing of the network, only needs to be done once """
import preprocess

from forward import Forward
from params import time_stamp

""" Set up the parameters you want to change compared to default ones """

""" parameters: three situations allowed:
 i) no input, then default values will be used;
 ii) list of the same length as pids (must be in the same order as 
 ['trc', 'trp', 'b1', 'b2', 'b3', 'b4', 'F1cc', 'F1cp', 'F1pp', 'F2cc', 'F2cp', 'F2pp', 'F3cc', 'F3cp', 'F3pp', 'F4cc', 
 'F4cp', 'F4pp', 'z', 'z_city', 'z_length', 'z_pos', 'R0', 'DE', 'DI']),
    will replace parameters in pids 
 iii) list of same length as fwd.para_id = 24, order must be respected """


##############################################################
""" reset parameters and do a forward run """

#pids =  ['trc', 'trp', 'b1', 'b2', 'b3', 'b4', 'F1cc', 'F1cp', 'F1pp', 'F2cc', 'F2cp', 'F2pp', 'F3cc', 'F3cp', 'F3pp', 'F4cc', 
# 'F4cp', 'F4pp', 'z', 'z_city', 'z_length', 'z_pos', 'R0', 'DE', 'DI']

""" Initialise the Forward object, positive flag: do not compute gradients, fast; negative flag: compuate gradients w.r.t.
 pids, the more parameters to compute the more time needed, takes a ~0.5min with all paramters included for a run. """
#fwd = Forward(1, pids)
#x0 = [0.4, 0.05, 0.2, 0.5, 0.5, 2.0, 1.000, 0.500, 0., 2.000, 2.000, 0.500, 0.100, 0.100, 0.100, 0, 3.000, 1.000, np.array([27.]), [8], np.array([[0,23]]), 'l', 2.68, 6.4, 3.]

#fwd.params['simu_range'] = 60           # simulation time range
#fwd.simulate(x0)

###############################################################
start_time = time.time()
print('[Start]')

ncity = 347
z_city_pool = [i for i in range(ncity)]

result_matrix_s = [[]]*ncity
result_matrix_e = [[]]*ncity
result_matrix_i = [[]]*ncity
result_matrix_r = [[]]*ncity

#pids =  ['trc', 'trp', 'b1', 'b2', 'b3', 'b4', 'F1cc', 'F1cp', 'F1pp', 'F2cc', 'F2cp', 'F2pp', 'F3cc', 'F3cp', 'F3pp', 'F4cc', 
# 'F4cp', 'F4pp', 'z', 'z_city', 'z_length', 'z_pos', 'R0', 'DE', 'DI']
#fwd = Forward(1, pids)
#fwd.params['simu_range'] = 30           # simulation time range
#x = [0.4, 0.05, 0.2, 0.5, 0.5, 2.0, 1.000, 0.500, 0., 2.000, 2.000, 0.500, 0.100, 0.100, 0.100, 0, 3.000, 1.000, np.array([27.]),
#                    [8], np.array([[0,23]]), 'l', 2.68, 6.4, 3.]

pids = ['z_city']
fwd = Forward(1, pids)
fwd.params['simu_range'] = 30     

z_sum = fwd.params['zzz'][0][0]*fwd.params['zzz'][1][0]
#z_sum = 1

# Write flowmap #
f_in = fwd.total_flow_in
f_out = fwd.total_flow_out

df = pd.DataFrame(f_in)
df.to_csv('output/' + str(time_stamp) + '_Flowmap_in.csv')
df = pd.DataFrame(f_out)
df.to_csv('output/' + str(time_stamp) + '_Flowmap_out.csv')

for city in range(len(z_city_pool)):
    
    if city%50 == 0:
        print('Progress ' + str(city) + ' / ' + str(len(z_city_pool)))
        
    fwd.simulate([[z_city_pool[city]]])
    
    # Collect results #
    result_matrix_s[city] = [k/z_sum for k in fwd.ss[-1,:]]
    result_matrix_e[city] = [k/z_sum for k in fwd.ee[-1,:]]
    result_matrix_i[city] = [k/z_sum for k in fwd.ii[-1,:]]
    result_matrix_r[city] = [k/z_sum for k in fwd.rr[-1,:]]
        
    # Write series #
    #names = ['S', 'E', 'I', 'R']
    #for i, f in enumerate([fwd.ss, fwd.ee, fwd.ii, fwd.rr]):
    #    df = pd.DataFrame(f)
    #    df.to_csv('output/' + str(int(time.time())) + '_Series_' + names[i] +'.csv')

# Write results #
names = ['S', 'E', 'I', 'R']
for i, f in enumerate([result_matrix_s, result_matrix_e, result_matrix_i, result_matrix_r]):
    df = pd.DataFrame(f)
    df.to_csv('output/' + str(time_stamp) + '_Matrix_' + names[i] +'.csv')

print('[End]')
print('Elapsed Time: ' + str(round(time.time()-start_time)) + ' s.')