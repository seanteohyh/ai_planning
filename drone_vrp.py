from ast import Continue
from asyncio.format_helpers import _format_callback_source
from calendar import c
import configparser
import json
import copy 
from math import inf
from scipy.spatial.distance import cdist, pdist
import numpy as np
import random


class Parser(object):
    def __init__(self, config_file, data_type):
        """Initialise parser
        Args: config_file::str
            path to config file
        """
        parser = configparser.ConfigParser()
        parser.read(config_file)
        self.config = parser[data_type]
        self.warehouses = []
        self.customers = []
        self.trucks = []
        self.drones = []
        self.map_size = json.loads(self.config['map_size'])
              
        self.ini_nodes()
        self.set_trucks()
        self.set_drones()
        
    def ini_nodes(self):
        for node in json.loads(self.config['nodes']):
            if int(node['type']) == 0:
                self.warehouses.append(Warehouse(int(node['id']), int(node['type']), 
                                   int(node['x']), int(node['y'])))
            elif int(node['type']) == 1:
                self.customers.append(Customer(int(node['id']), int(node['type']), 
                                   int(node['x']), int(node['y']), float(node['demand'])))
    def set_trucks(self):
        for truck in json.loads(self.config['trucks']):
            self.trucks.append(Truck(int(truck['id']), self.warehouses[int(truck['warehouse'])], 
                                   int(truck['speed_factor'])))
            
    def set_drones(self):
        for drone in json.loads(self.config['drones']):
            self.drones.append(Drone(int(drone['id']), self.warehouses[int(drone['warehouse'])], 
                                   int(drone['speed_factor']), int(drone['item_capacity']), int(drone['battery_capacity']),
                                   int(drone['consumption_rate']), int(drone['charging_speed'])))
        
        
class Point(object):

    def __init__(self, x, y):
        '''Initialize a point
        Args:
            x::int
                x coordinate of the point
            y::int
                y coordinate of the point
        '''
        self.x = x
        self.y = y

    def __str__(self):
        return 'Point, x: {}, y: {}'.format(self.x, self.y)

class Node(Point):
    
    def __init__(self, id, type, x, y):
        '''Initialize a node
        Args:
            id::int
                id of the node
            type::int
                0 for warehouses, 1 for customer
            x::int
                x coordinate of the node
            y::int
                y coordinate of the node
        '''
        super(Node, self).__init__(x, y)
        self.id = id
        self.type = type
    
    
    def get_nearest_node(self, nodes):
        '''Find the nearest node in the list of nodes
        Args:
            nodes::[Node]
                a list of nodes
        Returns:
            node::Node
                the nearest node found
        '''
        dis = [cdist([[self.x, self.y]], [[node.x, node.y]], 'cityblock') for node in nodes]
        idx = np.argmin(dis)
        return nodes[idx]

    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}'.format(self.id, self.type, self.x, self.y)
    
    
class Warehouse(Node):
    
    def __init__(self, id, type, x, y):
        '''Initialize a warehouse,
        Args:
            id::int
                id of the warehouse
            type::int
                0 for warehouse
            x::int
                x coordinate of the warehouse
            y::int
                y coordinate of the warehouse
        '''
        super(Warehouse, self).__init__(id, type, x, y)

            
    def replenish_items(self, vehicle, x):
        '''Replenish drone items to item level x
        Args:
            vehicle:: truck/drone object to receive items
            x::int 
                target inventory level to fulfill to
        '''
        
        assert(x<= vehicle.item_capacity)  
        vehicle.item = x

        
        
class Customer(Node):
    
    def __init__(self, id, type, x, y, demand):
        '''Initialize a customer
        Args:
            id::int
                id of the customer
            type::int
                1 for customer
            x::int
                x coordinate of the customer
            y::int
                y coordinate of the customer
            demand::int
                demand of the customer
        '''
        super(Customer, self).__init__(id, type, x, y)
        self.demand = demand
        self.turn_served = 1e3
        self.serve_by_truck = False
        
    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}, demand: {}, turn_served: {}'.format(self.id, self.type, self.x, self.y, self.demand, self.turn_served)
        
        
class Vehicle(object):
    
    def __init__(self, id, start_node, speed_factor):
        ''' Initialize the vehicle
        Args:
            id::int
                id of the vehicle
            start_node::Node
                starting node of the vehicle
            speed_factor::int
                speed factor of the vehicle
        '''
        self.id = id
        self.start_node = start_node
        self.speed_factor = speed_factor
        # travel time of the vehicle
        self.travel_turn = 0
        # all the (points, time) including warehouse, customers, or waiting point visited by the vehicle::[(x1,y1,t1), (x2,y2,t2)]
        self.visited_points = [(self.start_node.x, self.start_node.y, 0)] # start from warehouse

    def time_to_point(self, dest):
        '''Get time required for truck to reach a customer
        Args:
            dest:: target object
                    target customer/wh for vehicle to go to

        '''
        checker_vehicle= copy.deepcopy(self)
        checker_vehicle.travel_to(dest, 0)
        return checker_vehicle.visited_points[-1][2]

    def serve_customer(self, customer):
        '''gives customer quantity of items demanded
        Args:
            customer::Customer object
        '''
        current_x, current_y, current_t = self.visited_points[-1]
        assert(self.items >= customer.demand)
        assert(current_x == customer.x)
        assert(current_y == customer.y)
        self.items -= customer.demand
        customer.turn_served = current_t  
        
        
class Truck(Vehicle):
    
    def __init__(self, id, start_node, speed_factor):
        ''' Initialize a truck
        Args:
            id::int
                id of the truck
            start_node::Node
                starting node of the truck
            speed_factor::int
                speed factor of the truck
        '''
        super(Truck, self).__init__(id, start_node, speed_factor)
        self.half_turn = True
        self.items = 1e9

    def find_point_at_t(self, t):
        '''Returns the (x,y) coordinate at time t
        Args:
            t::int
                the time corresponding to the point you are looking for
        Returns:
            (x, y)::tuple(int, int)
        '''
        for p in self.visited_points:
            if p[2] == t:
                x = p[0]
                y = p[1]
                break
        return (x, y)

    def travel_x(self, point):
        '''Travel horizontally till self.x matches point.x, and updates attributes
        Args:
            point::Point object
                a point or customer or warehouse destination 
        '''
        current_x, current_y, current_t = self.visited_points[-1]
        while current_x != point.x:
            if self.half_turn:
                current_x += 1 if current_x < point.x else -1
                self.half_turn = False
            else:
                self.half_turn = True
            current_t += 1
            self.visited_points.append((current_x, current_y, current_t))
            self.travel_turn += 1

    def travel_y(self, point):
        '''Travel vertically till self.y matches point.y, and updates attributes
        Args:
            point::Point object
                a point or customer or warehouse destination
        '''
        current_x, current_y, current_t = self.visited_points[-1]
        while current_y != point.y:
            if self.half_turn:
                current_y += 1 if current_y < point.y else -1
                self.half_turn = False
            else:
                self.half_turn = True
            current_t += 1
            self.visited_points.append((current_x, current_y, current_t))
            self.travel_turn += 1

    def travel_to(self, point, vertical_first=True):
        '''Travel to indicated point
        Args:
            point::Point object
                a point or customer or warehouse destination
            vertical_first::boolean
                1 to indicate travel_y first then travel_x, 0 otherwise
        '''
        if vertical_first:
            self.travel_y(point)
            self.travel_x(point)
        else:
            self.travel_x(point)
            self.travel_y(point)

    def charge_to(self, drone, x):
        '''Charge drone to battery level x
        Args:
            drone::drone object
                the drone to charge
            x::int
                target battery level to charge to
        '''
        drone.on_truck = True
        while drone.battery_level < x:
            drone.battery_level += drone.charging_speed
            if drone.battery_level > drone.battery_capacity:
                drone.battery_level = drone.battery_capacity
            _, _, current_t = drone.visited_points[-1]
            px, py = self.find_point_at_t(current_t+1)
            drone.travel_on_truck(Point(px, py), current_t+1)
        drone.on_truck = False

    def replenish_drone(self, drone, x):
        '''Replenish drone items to item level x
        Args:
            drone:: drone object
                the drone to charge
            x::int 
                target inventory level to replenish to 
        '''
        
        assert(x <= drone.item_capacity)
        if drone.items >= x:
            amt = 0
        else:
            amt = x - drone.items
        drone.items += amt
    

    def wait(self, x, t):
        '''at turn t, wait at current point for x turns
        Args:
            x::int
                the number of turns to wait
        '''
        i = len(self.visited_points)-1
        current_x, current_y, current_t = self.visited_points[i]
        while current_t != t:
            self.visited_points[i][2] += x
            i -= 1
            current_x, current_y, current_t = self.visited_points[i]                 
        while x > 0:
            x -= 1
            i += 1
            current_t += 1
            self.visited_points.insert((current_x, current_y, current_t), i)


    def vert_hor(self, unvisited_lst, dest, drone):

        '''compare to get to dest from current position whether better to go vertical or horizontal first (get number of nearby customers from unvisited_lst)
        Args:
            unvisited_lst:: list
                    list of remaining customers unvisited
            dest:: customer object
                    target customer for truck to go to
            drone:: drone object
                    drone reference for battery capacity  (IF CAN SOMEHOW JUST GET THE VALUE INSTEAD OF HAVING TO PUT WOULD BE BETTER NOT SURE HOW)
        
        
        '''
        start_x = self.visited_points[-1][0]
        start_y = self.visited_points[-1][1]
        #horizontal first 
        hor_lst = []
        i = 0
        while (start_x+i) != dest.x:
            hor_lst.append((start_x+i, start_y))
            i += 1 if start_x+i < dest.x else -1

        j = 0
        while (start_y +j) != dest.y:
            hor_lst.append((start_x +i, start_y +j))
            j += 1 if start_y +j < dest.y else -1  

        hor_ind = 0
        for cust in unvisited_lst:
            for pt in hor_lst:
                if max(abs(cust.x-pt[0]),abs(cust.y-pt[1])) <= drone.battery_capacity//2+1:
                    hor_ind += 1
                    break

        #vertical fist
        vert_lst = []
        j = 0
        while(start_y +j) != dest.y:
            vert_lst.append((start_x, start_y+j))
            j += 1 if start_y + j < dest.y else -1

        i = 0
        while (start_x+i) != dest.x:
            vert_lst.append((start_x+i, start_y+j))
            i +=1 if start_x+i < dest.x else -1
        
        vert_ind = 0
        for cust in unvisited_lst:
            for pt in vert_lst:
                if max(abs(cust.x-pt[0]),abs(cust.y-pt[1])) <= drone.battery_capacity//2+1:
                    vert_ind += 1
                    break
                    
        return 0 if hor_ind > vert_ind else 1


    def __str__(self):
        return 'Truck id: {}, visited_points: {}'.format(self.id, self.visited_points)
        
        
class Drone(Vehicle):
   
    def __init__(self, id, start_node, speed_factor, item_capacity, battery_capacity, consumption_rate, charging_speed):
        ''' Initialize a drone
        Args:
            id::int
                id of the drone
            start_node::Node
                starting node of the drone
            speed_factor::int
                speed factor of the drone
            item_capacity::int
                item capacity of the drone
            battery_capacity::int
                 battery capacity of the drone   
            consumption_rate::int
                 consumption rate of the drone: how much battery level consumed per grid moved
            charging_speed::int
                 charging speed of the drone: how much battery level recovered per turn charged
        '''
   
        super(Drone, self).__init__(id, start_node, speed_factor) 
        self.item_capacity = item_capacity
        self.items = item_capacity
        self.battery_capacity = battery_capacity
        self.consumption_rate = consumption_rate
        self.charging_speed = charging_speed
        self.battery_level = battery_capacity
        self.on_truck = False

    def travel_straight(self, point):
        '''Travel vertically or horizontally till self.x or self.y matches point.x or point.y, OR till a direct diagonal path to point can be found
        Args:
            point::Point object
                a point or customer or warehouse destination
        '''
        current_x, current_y, current_t = self.visited_points[-1]
        direct_diag = (abs(point.x-current_x) == abs(point.y-current_y))
        while not direct_diag:
            if abs(point.x-current_x) > abs(point.y-current_y):
                current_x += 1 if current_x < point.x else -1
            else:
                current_y += 1 if current_y < point.y else -1     
            current_t += 1          
            self.visited_points.append((current_x, current_y, current_t))
            self.battery_level -= 1 if self.on_truck == False else 0
            self.travel_turn += 1
            direct_diag = (abs(point.x-current_x) == abs(point.y-current_y))

    def travel_diag(self, point):
        '''Travel diagonally till point is reached OR till self.x or self.y matches point.x or point.y
        Args:
            point::Point object
                a point or customer or warehouse destination
        '''
        current_x, current_y, current_t = self.visited_points[-1]
        direct_line = (point.x == current_x or point.y == current_y)
        while not direct_line:
            current_x += 1 if current_x < point.x else -1
            current_y += 1 if current_y < point.y else -1
            current_t += 1          
            self.visited_points.append((current_x, current_y, current_t))
            self.battery_level -= 1 if self.on_truck == False else 0
            self.travel_turn += 1
            direct_line = (point.x == current_x or point.y == current_y)

    def travel_to(self, point, diagonal_first=True):
        '''Travel to indicated point
        Args:
            point::Point object
                a point or customer or warehouse destination
            diagonal_first::boolean
                1 to indicate travel_diag first then travel_straight, 0 otherwise
        '''
        if diagonal_first:
            self.travel_diag(point)
            self.travel_straight(point)
        else:
            self.travel_straight(point)
            self.travel_diag(point)
            
    def wait(self,point):
        '''Wait at indicated point till truck reaches point
        saved_point(x,y,t) from check_truck function
        '''
        current_x, current_y , current_t = self.visited_points[-1]    
        assert((current_x== point[0]) & (current_y == point[1])) 
    
        while current_t < point[2]:
            current_t += 1
            self.visited_points.append((current_x, current_y, current_t))

            
    def replenish_inve(self):
        '''Top up inventory. Function only to be use when at warehouse or truck
        '''
        self.items = self.item_capacity

    def travel_on_truck(self, point, t):
        '''Point travelled while charging on truck
        Args:
            point::Point object
                the point when drone finish charging
            t::int
                the turn when drone finish charging
        '''
        self.visited_points.append((point.x, point.y, t))

    def check_cust(self, customer):
        '''Check if route to customer is feasible
        Args:
            customer::Customer object
                the customer we want the drone to travel to
        Returns:
            CheckerDrone object
        '''
        checker = CheckerDrone(copy.deepcopy(self))      
        checker.travel_to_cust(customer)
        if checker.drone.items < customer.demand:
            checker.drone.battery_level = -1
        else:
            checker.drone.items -= customer.demand
        return checker

    def check_wh(self, warehouses):
        '''Check if route to any warehose is feasible
        Args:
            warehouses::List of warehouse objects
                the list of warehouses we want to check if drone route is feasible to
        Returns:
            CheckerDrone object
        '''
        min_turn = 1e3
        target_wh = None
        for wh in warehouses:
            checker = CheckerDrone(copy.deepcopy(self))
            checker.travel_to_wh(wh)
            if checker.drone.battery_level >= 0:
                if checker.drone.travel_turn < min_turn:
                    min_turn = checker.drone.travel_turn
                    target_wh = wh

            checker = CheckerDrone(copy.deepcopy(self)) 
            if target_wh == None:
                checker.drone.battery_level = -1
            else:
                checker.travel_to_wh(target_wh)
                target_wh.replenish_items(checker.drone, checker.drone.item_capacity)
            return checker

    def check_truck(self, trucks):
        '''Check if route to any trucks is feasible
        Args:
            trucks::List of truck objects
                the list of trucks we want to check if drone route is feasible to
        Returns:
            CheckerDrone object
        '''
        min_turn = 1e7
        target_truck = None
        target_point = None

        for truck in trucks:
            for point in truck.visited_points[:-1]:
                if point[2] <= self.travel_turn or point[2] > self.travel_turn+self.battery_level:
                    continue
                checker = CheckerDrone(copy.deepcopy(self)) 
                checker.travel_to_truck(Point(point[0], point[1]), truck, point[2])
                
                if checker.drone.visited_points[-1][2] <= point[2] and checker.drone.battery_level >= 0:
                    if checker.drone.travel_turn < min_turn:
                        min_turn = checker.drone.travel_turn
                        target_truck = truck 
                        target_point = Point(point[0], point[1])
                        target_waitime = (point[0], point[1], point[2])

        checker = CheckerDrone(copy.deepcopy(self))  
        if target_truck == None:
            checker.drone.battery_level = -1
        else:
            checker.travel_to_truck(target_point, target_truck, target_waitime[2])
            checker.drone.wait(target_waitime)
            target_truck.charge_to(checker.drone, checker.drone.battery_capacity)
            target_truck.replenish_drone(checker.drone, checker.drone.item_capacity)
        return checker



    def __str__(self):
        return 'Drone id: {}, battery_level: {}, visited_points: {}'\
            .format(self.id, self.battery_level, self.visited_points)

class CheckerDrone(object):

    target_cust = None
    target_wh = None
    target_truck = None
    target_truck_location = (None, None, None)

    def __init__(self, drone):
        self.drone = drone

    def evaluate(self):
        if self.drone.battery_level < 0:
            return False
        return True

    def travel_to_cust(self, customer):
        self.drone.travel_to(customer, diagonal_first=True)
        self.target_cust = customer
    
    def travel_to_wh(self, wh):
        self.drone.travel_to(wh, diagonal_first=True)
        self.target_wh = wh

    def travel_to_truck(self, point, truck, truck_t):
        self.drone.travel_to(point, diagonal_first=True)
        x, y, _ = self.drone.visited_points[-1]
        self.target_truck = truck
        self.target_truck_location = (x, y, truck_t)

    def check_cust(self, customer):
        checker = self.drone.check_cust(customer)
        self.pass_down(checker)
        return checker

    def check_wh(self, warehouses):
        checker = self.drone.check_wh(warehouses)
        self.pass_down(checker)
        return checker

    def check_truck(self, trucks):
        checker = self.drone.check_truck(trucks)
        self.pass_down(checker)
        return checker

    def pass_down(self, child):
        if child.target_cust == None:
            child.target_cust = self.target_cust
        if child.target_wh == None:
            child.target_wh = self.target_wh
        if child.target_truck == None:
            child.target_truck = self.target_truck
        child.target_truck_location = self.target_truck_location  
        

    
        
class DVRP(object):
    
    def __init__(self, warehouses, customers, trucks, drones, map_size):
        '''Initialize the DVRP state
        Args:
            warehouses::[Warehouses]
                warehouses of the instance
            customers::[Customer]
                customers of the instance
            trucks::[Trucks]
                trucks of the instance
            drones::[Drones]
                drones of the instance
            map_size::[Drones]
                drones of the instance    
        '''

        self.warehouses = warehouses
        self.customers = customers
        self.trucks = trucks
        self.drones = drones
        self.map_size = map_size
        
        self.destroyed_nodes = []
        self.destroyed_idx = []


    def split_route(self):
        ''' Future Initialization for construction heuristic'''
        self.restart()
        for c in range(len(self.customers)):
            cust = self.customers[c]


            # drone checks
            best_drone_time = 1e3
            best_mtd = 0
            
            for d in range(len(self.drones)):
                drone = self.drones[d]
                if cust.demand > drone.item_capacity:
                    break

                drone_time = 1e3
                # 1. drone cust-truck check
                if drone.check_cust(cust).check_truck(self.trucks).evaluate():
                    test_drone = drone.check_cust(cust)
                    drone_time = test_drone.drone.visited_points[-1][2]
                    mtd = 1
                    
                # 2. drone- warehouse - cust- truck check  
                elif drone.check_wh(self.warehouses).check_cust(cust).check_truck(self.trucks).evaluate():
                    test_drone = drone.check_wh(self.warehouses).check_cust(cust)
                    drone_time = test_drone.drone.visited_points[-1][2]
                    mtd = 2

                # 3. drone - truck - cust- truck check
                elif drone.check_truck(self.trucks).check_cust(cust).check_truck(self.trucks).evaluate():
                    test_drone = drone.check_truck(self.trucks).check_cust(cust)
                    trgt_truck = test_drone.target_truck
                    drone_time = test_drone.drone.visited_points[-1][2]
                    mtd = 3
                
                if drone_time < best_drone_time:
                    best_drone_time = drone_time
                    best_check = test_drone
                    best_drone = drone
                    best_mtd=  mtd
                    
            if best_drone_time < 1e3:
                if best_mtd == 1:
                    best_drone.travel_to(best_check.target_cust)
                    best_drone.serve_customer(best_check.target_cust)
                if best_mtd == 2:
                    best_drone.travel_to(best_check.target_wh)
                    best_drone.replenish_inve()
                    best_drone.travel_to(best_check.target_cust)
                    best_drone.serve_customer(best_check.target_cust)
                if best_mtd == 3:
                    best_drone.travel_to(Point(best_check.target_truck_location[0],best_check.target_truck_location[1]))
                    best_drone.wait(best_check.target_truck_location)
                    best_check.target_truck.charge_to(best_drone, best_drone.battery_capacity)
                    best_drone.replenish_inve()
                    best_drone.travel_to(best_check.target_cust)
                    best_drone.serve_customer(best_check.target_cust)                   
                continue
            
            # truck checks    
            best_truck_time = 1e7
            for truck in self.trucks:
                
                time = truck.time_to_point(cust)
                if time < best_truck_time:
                    best_truck_time = time
                    best_truck = truck
             
            direction = best_truck.vert_hor(self.customers[c:], cust, self.drones[0])
            best_truck.travel_to(cust,direction)
            best_truck.serve_customer(cust)

    def restart(self):
        for c in self.customers:
            c.turn_served = 1e3
            c.serve_by_truck = False
        for d in self.drones:
            d.visited_points = [(d.start_node.x, d.start_node.y, 0)]
            d.battery_level = d.battery_capacity
            d.items = d.item_capacity
            d.on_truck = False
            d.travel_turn = 0
        for t in self.trucks:
            t.visited_points = [(t.start_node.x, t.start_node.y, 0)]
            t.half_turn = False
            t.travel_turn = 0
        self.destroyed_nodes = []

    def restart_drone(self):
        for c in self.customers:
            if not c.serve_by_truck:
                c.turn_served = 1e3
        for d in self.drones:
            d.visited_points = [(d.start_node.x, d.start_node.y, 0)]
            d.battery_level = d.battery_capacity
            d.items = d.item_capacity
            d.on_truck = False
            d.travel_turn = 0        

    def split_route_far(self):
        ''' Future Initialization for construction heuristic'''
        self.restart()

        while any([(c.turn_served==1e3) for c in self.customers]):
            for c in range(len(self.customers)):

                if self.customers[c].serve_by_truck:
                    continue

                cust = self.customers[c]

                # drone checks
                best_drone_time = 1e3
                best_mtd = 0
                
                for d in range(len(self.drones)):
                    drone = self.drones[d]
                    if cust.demand > drone.item_capacity:
                        break

                    drone_time = 1e3
                    # 1. drone cust-truck check
                    if drone.check_cust(cust).check_truck(self.trucks).evaluate():
                        test_drone = drone.check_cust(cust)
                        drone_time = test_drone.drone.visited_points[-1][2]
                        mtd = 1
                        
                    # 2. drone- warehouse - cust- truck check  
                    elif drone.check_wh(self.warehouses).check_cust(cust).check_truck(self.trucks).evaluate():
                        test_drone = drone.check_wh(self.warehouses).check_cust(cust)
                        drone_time = test_drone.drone.visited_points[-1][2]
                        mtd = 2

                    # 3. drone - truck - cust- truck check
                    elif drone.check_truck(self.trucks).check_cust(cust).check_truck(self.trucks).evaluate():
                        test_drone = drone.check_truck(self.trucks).check_cust(cust)
                        trgt_truck = test_drone.target_truck
                        drone_time = test_drone.drone.visited_points[-1][2]
                        mtd = 3
                    
                    if drone_time < best_drone_time:
                        best_drone_time = drone_time
                        best_check = test_drone
                        best_drone = drone
                        best_mtd=  mtd
                
                if best_drone_time < 1e3:
                    if best_mtd == 1:
                        best_drone.travel_to(best_check.target_cust)
                        best_drone.serve_customer(best_check.target_cust)
                    if best_mtd == 2:
                        best_drone.travel_to(best_check.target_wh)
                        best_drone.replenish_inve()
                        best_drone.travel_to(best_check.target_cust)
                        best_drone.serve_customer(best_check.target_cust)
                    if best_mtd == 3:
                        best_drone.travel_to(Point(best_check.target_truck_location[0],best_check.target_truck_location[1]))
                        best_drone.wait(best_check.target_truck_location)
                        best_check.target_truck.charge_to(best_drone, best_drone.battery_capacity)
                        best_drone.replenish_inve()
                        best_drone.travel_to(best_check.target_cust)
                        best_drone.serve_customer(best_check.target_cust)                   
                    continue
                
                # truck checks    
                best_truck_time = 1e7
                for truck in self.trucks:
                    
                    time = truck.time_to_point(cust)
                    if time < best_truck_time:
                        best_truck_time = time
                        best_truck = truck
                
                direction = best_truck.vert_hor(self.customers[c:], cust, self.drones[0])
                best_truck.travel_to(cust,direction)
                best_truck.serve_customer(cust)
                cust.serve_by_truck = True
                self.restart_drone()
                if c != len(self.customers)-1:
                    break
            
    def random_initialize(self, seed=None):
        ''' Randomly initialize the state with split_route() (your construction heuristic)
        Args:
            seed::int
                random seed
        Returns:
            objective::float
                objective value of the state
        '''
        if seed is not None:
            random.seed(606)
        random.shuffle(self.customers)
        self.split_route()
        return self.objective()
    
    def random_initialize_far(self, seed=None):
        ''' Randomly initialize the state with split_route() (your construction heuristic)
        Args:
            seed::int
                random seed
        Returns:
            objective::float
                objective value of the state
        '''
        if seed is not None:
            random.seed(606)
        random.shuffle(self.customers)
        self.split_route_far()
        return self.objective()
        
        
    def objective(self):
        ''' Calculate the objective value of the state
        Return turns needed to fulfil customer orders and for vehicles to travel back to warehouse
        '''
        return sum([c.turn_served for c in self.customers])


