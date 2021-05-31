#!/usr/bin/env python
# coding: utf-8
# coding: unicode_escape
## params ##

import numpy as np

time_stamp = 1614952602 
filename = "data/chinese_cities.csv"
filename_py = "data/chinese_cities_py.csv"
names_eng = ['Province', 'Name', 'Area', 'Population', 'Population Density', 'GDP', 'GDP year',
             'GDP per capita', 'Series in Province', 'City level', 'Population Flow 2018', 'Population Flow 2017',
             'Train 1', 'Train 2', 'Aero flow 50', 'Aero flow 100', 'Lat.', 'Lon.']
##### Run #####
x0 = [0.4, 0.05, 0.2, 0.5, 0.5, 2.0, 1.000, 0.500, 0., 2.000, 2.000, 0.500, 0.100, 0.100, 0.100, 0, 3.000, 1.000, np.array([5]), [8], np.array([[0,1]]), 'l', 2.5, 6, 3]

x0 = [0.4, 0.05, 0.2, 0.5, 0.5, 2.0, 2.000, 1.000, 0., 4.000, 4.000, 1.000, 0.200, 0.200, 0.200, 0, 6.000, 2.000, np.array([5]), [8], np.array([[0,1]]), 'l', 2.5, 6, 3]

##### EpiRank #####
#alpha_pool = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
alpha_pool = [0,0.25,0.5,0.75,1]
alpha = 0.5
popu_cutoff = 100

##### Visualization #####
visual_cutoff = 0.98
title_1 = "中国地级行政区传染病危害指数 \nChinese City Epidemic Hazard Index"
title_2 = "tianyil@mit.edu "
title_2 += "\n"
title_2 += "jiawen.luo@erdw.ethz.ch"


##### System #####
timeout_max = 86400
params_name = "params.py"
command1 = "python3 run.py"
command2 = "python3 EpiRank.py"
