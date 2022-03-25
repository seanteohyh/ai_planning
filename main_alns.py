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
import matplotlib.animation as animation


def draw_animated_output(dvrp):
    width = dvrp.map_size[0]
    height = dvrp.map_size[1]
    fig = plt.figure() 
    ax = plt.axes(xlim=(-5, width+5), ylim=(-5, height+5))
    max_frames = max([c.turn_served for c in dvrp.customers])+2
    
    # scatter plot for warehouses
    wh = [(wh.x, wh.y) for wh in dvrp.warehouses]
    wh_x = [i[0] for i in wh]
    wh_y = [i[1] for i in wh]
    plt.scatter(
        wh_x, 
        wh_y, 
        c ="yellow", 
        linewidths = 2,
        marker ="^",
        edgecolor ="red",
        s = 200
    )

    # scatter plot for customers
    c = [(c.x, c.y, c.id) for c in dvrp.customers]
    c_x = [i[0] for i in c]
    c_y = [i[1] for i in c]
    plt.scatter(
        c_x, 
        c_y, 
        c ="yellow", 
        linewidths = 2,
        marker ="s",
        edgecolor ="green",
        s = 200
    )    
    

    # lines for trucks and drones
    lines = []
    for _ in range(len(dvrp.drones)):
        lobj = ax.plot([],[],lw=2,color="blue")[0]
        lines.append(lobj)
    for _ in range(len(dvrp.trucks)):
        lobj = ax.plot([],[],lw=2,color="red")[0]
        lines.append(lobj)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    
    def animate_drones(i):
        index = 0
        target_t = i
        for drone in dvrp.drones:
            x_list = []
            y_list = []
            for t in range(target_t):
                if t < len(drone.visited_points):
                    x, y = [(p[0], p[1]) for p in drone.visited_points if p[2] == t][0]
                else:
                    x, y = drone.visited_points[-1][0], drone.visited_points[-1][1] 
                x_list.append(x)
                y_list.append(y)
            lines[index].set_data(x_list,y_list)
            index += 1
        for truck in dvrp.trucks:
            x_list = []
            y_list = []
            for t in range(target_t):
                if t < len(truck.visited_points):
                    x, y = [(p[0], p[1]) for p in truck.visited_points if p[2] == t][0]
                else:
                    x, y = truck.visited_points[-1][0], truck.visited_points[-1][1]
                x_list.append(x)
                y_list.append(y)
            lines[index].set_data(x_list,y_list)
            index += 1
        return lines

    anim = animation.FuncAnimation(fig, animate_drones, init_func=init, 
							frames=max_frames, interval=50, blit=True)
    
    # %matplotlib gt
    plt.show() 


def randomDestroy(current, random_state):
    ''' randomly destroys 20% of customer nodes. Remove 1 only if there are less than 5 customers.
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
    destroyed = copy.deepcopy(current)
    if round(len(destroyed.customers)) < 5:
        idx = random_state.randint(0,len(destroyed.customers))

        destroyed.destroyed_nodes.append(destroyed.customers.pop(idx))
        destroyed.destroyed_idx.append(idx)
    else:
        for _ in range(round(len(destroyed.customers)*0.2)):
            idx = random_state.randint(0,len(destroyed.customers))

            destroyed.destroyed_nodes.append(destroyed.customers.pop(idx))
            destroyed.destroyed_idx.append(idx)
    return destroyed

def greedyDestroy(current, random_state):
    ''' Greedily destroys the customer which gives the biggest reduction in total distance of the perturbed tour
    Args:
        current::DVRP
            an DVRP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        destroyed::DVRP
            the evrp object after destroying
    '''
    destroyed = copy.deepcopy(current)
    max_dist = -1
    for i in range(0,len(destroyed.customers)):
        i_minus_1 = (i-1)%len(destroyed.customers)
        i_plus_1 = (i+1)%len(destroyed.customers)
        distance = (
            ((destroyed.customers[i].x - destroyed.customers[i_minus_1].x)**2 + (destroyed.customers[i].y - destroyed.customers[i_minus_1].y)**2)**0.5 +
            ((destroyed.customers[i].x - destroyed.customers[i_plus_1].x)**2 + (destroyed.customers[i].y - destroyed.customers[i_plus_1].y)**2)**0.5 - 
            ((destroyed.customers[i_minus_1].x - destroyed.customers[i_plus_1].x)**2 + (destroyed.customers[i_minus_1].y - destroyed.customers[i_plus_1].y)**2)**0.5
        )
        if distance > max_dist:
            max_dist = distance
            target = i
    destroyed.destroyed_idx.append(target)
    destroyed.destroyed_nodes.append(destroyed.customers.pop(target))
    return destroyed    


# define worse destory action
def WorseDestroy(current, random_state):
    scores={}
    curr_score = current.objective()
    
    for c in current.customers:
        dvrp_ = copy.deepcopy(current)
        dvrp_.customers.remove([cust for cust in dvrp_.customers if cust.id == c.id][0])
        dvrp_.split_route()
        scores[c.id] = curr_score - dvrp_.objective()
        
    sorted_id = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    d = int(random.uniform(1, len(current.customers)/5))
    worse_n =  [k for k in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:d]]
    rm_list = [c[0] for c in worse_n]
    
    destroyed = current

    for c_id in rm_list:
        destroyed_cust = [c for c in destroyed.customers if c.id == c_id][0]
        current.destroyed_nodes.append(destroyed_cust)
        destroyed.customers.remove(destroyed_cust)
    
    return destroyed


### Repair operators ###
# You can follow the example and implement repair_2, repair_3, etc
def randomRepair(destroyed, random_state):
    ''' randomly inserts destroyed nodes into existing customer sequence
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
    repaired = copy.deepcopy(destroyed)
    while repaired.destroyed_nodes:
        repaired.customers.insert(random_state.randint(0,len(repaired.customers)), repaired.destroyed_nodes.pop())
    repaired.split_route()
    return repaired


def greedyRepair(destroyed, random_state):
    ''' greedily inserts destroyed nodes into existing customer sequence such that we obtain the smallest increase in total distance of the perturbed tour
    Args:
        destroyed::DVRP
            an DVRP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::DVRP
            the evrp object after repairing
    '''
    repaired = copy.deepcopy(destroyed)
    for c in repaired.destroyed_nodes:
        max_dist = -1e9
        for i in range(0,len(repaired.customers)):
            i_plus_1 = (i+1)%len(repaired.customers)
            distance = (
                ((repaired.customers[i].x - repaired.customers[i_plus_1].x)**2 + (repaired.customers[i].y - repaired.customers[i_plus_1].y)**2)**0.5 -
                ((c.x - repaired.customers[i].x)**2 + (c.y - repaired.customers[i].y)**2)**0.5 -
                ((c.x - repaired.customers[i_plus_1].x)**2 + (c.y - repaired.customers[i_plus_1].y)**2)**0.5 
            )
            if distance > max_dist:
                target = i_plus_1
                max_dist = distance
        repaired.customers.insert(target, c)
    repaired.split_route()
    return repaired

def nearidxRepair(destroyed, random_state):
    '''  inserts to index -1 position, if index = 0 insert at back
    Args:
        destroyed::DVRP
            an DVRP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::DVRP
            the evrp object after repairing
    '''
    repaired = copy.deepcopy(destroyed)
    for c in range(len(repaired.destroyed_nodes)-1,-1,-1):
        if repaired.destroyed_idx[c] == 0:
            repaired.customers.insert(len(repaired.customers), repaired.destroyed_nodes.pop(c))
            repaired.destroyed_idx.pop(c)
        else:
            repaired.customers.insert(repaired.destroyed_idx.pop(c)-1, repaired.destroyed_nodes.pop(c))

    repaired.split_route()
    return repaired
                


if __name__ == '__main__':
    # instance file and random seed
    config_file = "config.ini"
    data_type = "data-complex"
    
    # # load data and random seed
    parsed = Parser(config_file, data_type)
    dvrp = DVRP(parsed.warehouses, parsed.customers, parsed.trucks, parsed.drones, parsed.map_size)
    
    # # ## start ##
    seed = 606
    dvrp.random_initialize(seed)
    initial_objective = dvrp.objective()

    # ALNS
    random_state = rnd.RandomState(seed)
    alns = ALNS(random_state)
    # add destroy
    alns.add_destroy_operator(randomDestroy)
    alns.add_destroy_operator(greedyDestroy)
    # add repair
    alns.add_repair_operator(randomRepair)
    alns.add_repair_operator(greedyRepair)
    alns.add_repair_operator(nearidxRepair)
    
    # run ALNS
    # select cirterion
    # criterion = HillClimbing()
    criterion = SimulatedAnnealing(10000, 100, 100)

    # assigning weights to methods
    omegas = [5.0, 3.0, 1.0, 0]
    lambda_ = 0.6
    result = alns.iterate(dvrp, omegas, lambda_, criterion,
                          iterations=5, collect_stats=True)

    # result
    solution = result.best_state
    objective = solution.objective()
    print('Initial objective is {}.'.format(initial_objective))    
    print('Best heuristic objective is {}.'.format(objective))
    draw_animated_output(solution)

    # for drone in solution.drones:
    #     print(drone)
        