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
    fig = plt.figure() 
    ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
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
    c = [(c.x, c.y) for c in dvrp.customers]
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
							frames=max_frames, interval=500, blit=True)

    plt.show() 

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

    warehouses = [
        Warehouse(
            id=0,
            type=0,
            x=0,
            y=0
        ),
        Warehouse(
            id=1,
            type=0,
            x=6,
            y=6
        ),
        Warehouse(
            id=2,
            type=0,
            x=-6,
            y=-6
        )
    ]

    customers = [
        Customer(
            id=0,
            type=1,
            x=2,
            y=2,
            demand=1
        ),
        Customer(
            id=1,
            type=1,
            x=1,
            y=1,
            demand=1
        ),
        Customer(
            id=2,
            type=1,
            x=2,
            y=1,
            demand=1
        )
    ]
    
    trucks = [
        Truck(
            id=0,
            start_node=warehouses[0],
            speed_factor=1,
            item_capacity=10
        )
    ]

    drones = [
        Drone(
            id=0,
            start_node=warehouses[0],
            speed_factor=1,
            item_capacity=2,
            battery_capacity=5,
            consumption_rate=1,
            charging_speed=5
        )
    ]

    dvrp = DVRP(warehouses=warehouses, customers=customers, trucks=trucks, drones=drones, map_size=100)

    # checker = drones[0].check_wh(dvrp.warehouses).check_cust(dvrp.customers[0]).check_truck(dvrp.trucks)
    # print(checker.evaluate())
    # print(checker.drone)

    dvrp.split_route()
    
    for drone in dvrp.drones:
        print(drone)

    for truck in dvrp.trucks:
        print(truck)

    draw_animated_output(dvrp)

    # # ## start ##
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