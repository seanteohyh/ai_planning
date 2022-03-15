# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:30:05 2022

@author: ivanl
"""

from numpy import diag
from drone_vrp import *

if __name__ == '__main__':
    # instance file and random seed
    config_file = "config.ini"
    data_type = "DEFAULT"
    
    # # load data and random seed
    # parsed = Parser(config_file, data_type)
    
    # dvrp = DVRP(parsed.warehouses, parsed.customers, parsed.trucks, parsed.drones)
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

    customer0 = Customer(
        id=0,
        type=1,
        x=3,
        y=3,
        demand=2
    )

    customer1 = Customer(
    id=1,
    type=1,
    x=2,
    y=2,
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

    print(truck0)
    truck0.travel_to(Point(3,3), vertical_first=True)
    print(truck0)

    print(drone0)
    drone0.travel_to(Point(2,2), diagonal_first=True)
    print(drone0)
    drone0.serve_customer(customer1)
    print(drone0.items)
    drone0.travel_to(Point(1,3), diagonal_first=True)
    print(drone0)



    # Now truck and drone at same point
    truck0.charge_to(drone0,5)
    truck0.replenish_drone(drone0,1)
    print(drone0.items)

    truck0.serve_customer(customer0)
    print(truck0.items)



