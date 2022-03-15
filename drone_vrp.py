import configparser
import json


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
        # travel time of the vehicle
        self.travel_turn = 0
        # all the (points, time) including warehouse, customers, or waiting point visited by the vehicle::[(x1,y1,t1), (x2,y2,t2)]
        self.visited_points = [(self.start_node.x, self.start_node.y, 0)] # start from warehouse
        
        
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


    def __str__(self):
        return 'Truck id: {}, start_node: {}, speed_factor: {}, item_capacity: {}, visited_points: {}'.format(self.id, self.start_node.id, self.speed_factor, self.item_capacity, self.visited_points)
        
        
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

    def __str__(self):
        return 'Drone id: {}, start_node: {}, speed_factor: {}, item_capacity: {}, visited_points: {}, battery_capacity: {}, consumption_rate: {}, charging_speed: {}'\
            .format(self.id, self.start_node.id, self.speed_factor, self.item_capacity, self.visited_points, self.battery_capacity, self.consumption_rate, self.charging_speed)
    
        
class DVRP(object):
    
    def __init__(self, warehouses, customers, trucks, drones):
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
        '''

        self.warehouses = warehouses
        self.customers = customers
        self.trucks = trucks
        self.drones = drones