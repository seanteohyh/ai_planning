# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:30:05 2022

@author: ivanl
"""

from drone_vrp import *

if __name__ == '__main__':
    # instance file and random seed
    config_file = "config.ini"
    data_type = "DEFAULT"
    
    # load data and random seed
    parsed = Parser(config_file, data_type)
    
    dvrp = DVRP(parsed.warehouses, parsed.customers, parsed.trucks, parsed.drones)
    for w in dvrp.warehouses:
        print(w)
    for c in dvrp.customers:
        print(c)
    for t in dvrp.trucks:
        print(t)
    for d in dvrp.drones:
        print(d)