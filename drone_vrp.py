import configparser
import json
import copy
from scipy.spatial.distance import cdist, pdist
import numpy as np



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
                                   int(truck['speed_factor']), int(truck['item_capacity'])))
            
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

            
    def replenish_item(vehicle,x):
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
        self.turn_served = 0
        
    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}, demand: {}, turn_served: {}'.format(self.id, self.type, self.x, self.y, self.demand, self.turn_served)
        
        
class Vehicle(object):
    
    def __init__(self, id, start_node, speed_factor, item_capacity):
        ''' Initialize the vehicle
        Args:
            id::int
                id of the vehicle
            start_node::Node
                starting node of the vehicle
            speed_factor::int
                speed factor of the vehicle
            item_capacity::int
                item capacity of the vehicle
        '''
        self.id = id
        self.start_node = start_node
        self.speed_factor = speed_factor
        self.item_capacity = item_capacity
        self.items = item_capacity
        # travel time of the vehicle
        self.travel_turn = 0
        # all the (points, time) including warehouse, customers, or waiting point visited by the vehicle::[(x1,y1,t1), (x2,y2,t2)]
        self.visited_points = [(self.start_node.x, self.start_node.y, 0)] # start from warehouse


    def serve_customer(self, customer):
        '''gives customer quantity of items demanded
        Args:
            customer::Customer object
        '''
        current_x, current_y, current_t = self.visited_points[len(self.visited_points)-1]
        assert(self.items >= customer.demand)
        assert(current_x == customer.x)
        assert(current_y == customer.y)
        self.items -= customer.demand
        customer.turn_served = current_t  
        
        
class Truck(Vehicle):
    
    def __init__(self, id, start_node, speed_factor, item_capacity):
        ''' Initialize a truck
        Args:
            id::int
                id of the truck
            start_node::Node
                starting node of the truck
            speed_factor::int
                speed factor of the truck
            item_capacity::int
                item capacity of the truck
        '''
        super(Truck, self).__init__(id, start_node, speed_factor, item_capacity)
        self.wait = False

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
        current_x, current_y, current_t = self.visited_points[len(self.visited_points)-1]
        while current_x != point.x:
            current_x += 1 if current_x < point.x else -1
            current_t += 1
            self.visited_points.append((current_x, current_y, current_t))
            self.travel_turn += 1

    def travel_y(self, point):
        '''Travel vertically till self.y matches point.y, and updates attributes
        Args:
            point::Point object
                a point or customer or warehouse destination
        '''
        current_x, current_y, current_t = self.visited_points[len(self.visited_points)-1]
        while current_y != point.y:
            current_y += 1 if current_y < point.y else -1
            current_t += 1
            self.visited_points.append((current_x, current_y, current_t))
            self.travel_turn += 1

    def travel_to(self, point, vertical_first):
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
            _, _, current_t = drone.visited_points[len(drone.visited_points)-1]
            px, py = self.find_point_at_t(current_t+1)
            drone.travel_to(Point(px, py), diagonal_first=True)
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
        assert(self.items >= amt)
        self.items -= amt
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
   
        super(Drone, self).__init__(id, start_node, speed_factor, item_capacity)        
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

    def travel_to(self, point, diagonal_first):
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

    def __str__(self):
        return 'Drone id: {}, battery_level: {}, visited_points: {}'\
            .format(self.id, self.battery_level, self.visited_points)
    
        
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
        
        # record the all the customers who have been visited by all the vehicles, eg. [Customer1, Customer2, ..., Customer7, Customer8]
        self.customer_visited = []
        # record the unvisited customers, eg. [Customer9, Customer10]
        self.customer_unvisited = copy.deepcopy(customers)
        
    def initialize(self):
        ''' Initialize the state with construction heuristic
        Evenly distribute customers to each truck if sufficient capacity, restock otherwise
        Truck returns to starting warehouse after all customers served
        Returns:
            objective::float
                objective value of the state
        '''
        while self.customer_unvisited != []:
            for t in self.trucks:
                cust = self.customer_unvisited[0]
                if t.items >= cust.demand:
                    t.travel_to(Point(cust.x, cust.y), False)
                    t.serve_customer(cust)
                    self.customer_visited.append(self.customer_unvisited.remove(cust))
                else:
                    t.travel_to(Point(t.start_node.x, t.start_node.y), False)
                    t.items = t.item_capacity
        for t in self.trucks:
            t.travel_to(Point(t.start_node.x, t.start_node.y), False)
        return self.objective()
        
    def objective(self):
        ''' Calculate the objective value of the state
        Return turns needed to fulfil customer orders and for vehicles to travel back to warehouse
        '''
        return max([len(t.visited_points) for t in self.trucks] + [len(d.visited_points) for d in self.drones])
