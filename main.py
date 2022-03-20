# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:30:05 2022

@author: ivanl
"""

from numpy import diag
from drone_vrp import *
import sys
sys.path.append('./ALNS')
from alns import ALNS, State
from alns.criteria import HillClimbing, SimulatedAnnealing, RecordToRecordTravel
import numpy.random as rnd
import copy
import matplotlib.pyplot as plt
import numpy as np

def draw_output(dvrp, turn=0):
    width = dvrp.map_size[0]
    height = dvrp.map_size[1]
    
    board = np.zeros((width, height, 3))
    board += [1.0, 1.0, 1.0] # "Black" color. Can also be a sequence of r,g,b with values 0-1.
    board[::2, ::2] = [0.0, 0.8, 0.8] # "White" color
    board[1::2, 1::2] = [0.0, 0.8, 0.8] # "White" color
    
    fig, ax = plt.subplots()
    ax.imshow(board, interpolation='nearest')
    
    for w in dvrp.warehouses:
        ax.text(w.x, w.y, u'\N{house}', size=15, ha='center', va='center')
        
    for c in dvrp.customers:
        ax.text(c.x, c.y, '😀', size=15, ha='center', va='center')
        
    for t in dvrp.trucks:
        if len(t.visited_points) > turn:
            ax.text(t.visited_points[turn][0], t.visited_points[turn][1], "🚗", size=15, ha='center', va='center')
    
    for d in dvrp.drones:
        if len(d.visited_points) > turn:
            ax.text(d.visited_points[turn][0], d.visited_points[turn][1], "drone", size=15, ha='center', va='center') 
    
    ax.set(xticks=[], yticks=[])
    ax.axis('image')
    
    plt.show()


def destroy_1(current, random_state):
    ''' Destroy operator sample (name of the function is free to change)
    Args:
        current::DVRP
            an DVRP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        destroyed::DVRP
            the evrp object after destroying
    '''
    # You should code here
    destroyed = current
    return destroyed

### Repair operators ###
# You can follow the example and implement repair_2, repair_3, etc
def repair_1(destroyed, random_state):
    ''' repair operator sample (name of the function is free to change)
    Args:
        destroyed::DVRP
            an DVRP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::DVRP
            the evrp object after repairing
    '''
    # You should code here
    repaired = destroyed
    return repaired


if __name__ == '__main__':
    # instance file and random seed
    config_file = "config.ini"
    data_type = "DEFAULT"
    
    # # load data and random seed
    parsed = Parser(config_file, data_type)
    
    dvrp = DVRP(parsed.warehouses, parsed.customers, parsed.trucks, parsed.drones, parsed.map_size)
    # for w in dvrp.warehouses:
    #     print(w)
    # for c in dvrp.customers:
    #     print(c)
    # for t in dvrp.trucks:
    #     print(t)
    # for d in dvrp.drones:
    #     print(d)

    warehouse0 = Warehouse(
        id=0,
        type=0,
        x=0,
        y=0
    )

    warehouse1 = Warehouse(
        id=1,
        type=0,
        x=10,
        y=10
    )

    warehouse2 = Warehouse(
        id=1,
        type=0,
        x=-10,
        y=-10
    )

    customer0 = Customer(
        id=0,
        type=1,
        x=2,
        y=2,
        demand=2
    )

    customer1 = Customer(
        id=1,
        type=1,
        x=5,
        y=2,
        demand=1
    )

    customer2 = Customer(
        id=1,
        type=1,
        x=4,
        y=4,
        demand=1
    )
    
    truck0 = Truck(
        id=0,
        start_node=warehouse0,
        speed_factor=1,
        item_capacity=10
    )

    drone0 = Drone(
        id=0,
        start_node=warehouse0,
        speed_factor=1,
        item_capacity=1,
        battery_capacity=5,
        consumption_rate=1,
        charging_speed=5
    )

    print(drone0.check_wh([warehouse1, warehouse2])) # this will be false
    print(drone0.check_wh([warehouse0, warehouse1, warehouse2])) # this will be true
    
    print(drone0.check_cust(customer0)) # this will be true
    print(drone0.check_cust(customer0, consec_checks=True).check_truck(trucks=[truck0])) # this will be false

    truck0.travel_to(customer2, vertical_first=True)
    print(drone0.check_cust(customer0, consec_checks=True).check_truck(trucks=[truck0], save_points=True)) # now this will be true    
    
    # ## start ##
    # dvrp.initialize()
    # for t in dvrp.trucks:
    #     print(f"\n{t}")
    
    # for i in range(dvrp.objective()):
    #     draw_output(dvrp, i)
        
    # # ALNS
    # random_state = rnd.RandomState(606)
    # alns = ALNS(random_state)
    # # add destroy
    # alns.add_destroy_operator(destroy_1)
    # # add repair
    # alns.add_repair_operator(repair_1)
    
    # # run ALNS
    # # select cirterion
    # criterion = HillClimbing()
    # # assigning weights to methods
    # omegas = [3.0, 2.0, 1.0, 0]
    # lambda_ = 0.2
    # result = alns.iterate(dvrp, omegas, lambda_, criterion,
    #                       iterations=10, collect_stats=True)

    # # result
    # solution = result.best_state
    # objective = solution.objective()
    # print('Best heuristic objective is {}.'.format(objective))