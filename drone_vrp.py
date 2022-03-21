from ast import Continue
from asyncio.format_helpers import _format_callback_source
from calendar import c
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

            
    def replenish_item(self, vehicle, x):
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
        self.half_turn = True

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
            
    def wait(self,point):
        '''Wait at indicated point till truck reaches point
        saved_point(x,y,t) from check_truck function
        '''
        current_x, current_y , current_t = self.visited_points[-1]    
        assert(current_x== point[0] & current_y == point[1]) 
    
        while current_t < point[2]:
            current_t += 1
            self.visited_points.apped((current_x, current_y, current_t))
            
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

    def check_cust(self, customer, consec_checks=False):
        '''Check if route to customer is feasible
        Args:
            customer::Customer object
                the customer we want the drone to travel to
            consec_checks::Boolean
                True if we want to continue checking route to other points, False otherwise
        Returns:
            checker_drone if consec_checks is True, else 1 if feasible, 0 otherwise
        '''
        checker_drone = copy.deepcopy(self)      
        checker_drone.travel_to(customer, diagonal_first=True)

        if consec_checks:
            return checker_drone
        elif checker_drone.battery_level < 0:
            return False
        else:
            return True

    def check_wh(self, warehouses, consec_checks=False):
        '''Check if route to any warehose is feasible
        Args:
            warehouses::List of warehouse objects
                the list of warehouses we want to check if drone route is feasible to
            consec_checks::Boolean
                True if we want to continue checking route to other points, False otherwise
        Returns:
            checker_drone if consec_checks is True, else 1 if feasible, 0 otherwise
        '''
        min_turn = 1e3
        target_wh = None
        for wh in warehouses:
            checker_drone = copy.deepcopy(self) 
            checker_drone.travel_to(wh, diagonal_first=True)
            if checker_drone.battery_level >= 0:
                if not consec_checks:
                    return True
                if checker_drone.travel_turn < min_turn:
                    min_turn = checker_drone.travel_turn
                    target_wh = wh
        if not consec_checks:
            return False
        elif target_wh == None:
            return False
        else:
            checker_drone = copy.deepcopy(self) 
            checker_drone.travel_to(target_wh, diagonal_first=True)
<<<<<<< HEAD
            target_wh.replenish_items(checker_drone, checker_drone.item_capacity)
            print("WH CHECK", checker_drone.visited_points)
=======
>>>>>>> 5bc0894c96c2d541598d6ee8ec07dd85f834617f
            return checker_drone

    def check_truck(self, trucks, consec_checks=False, save_points=False):
        '''Check if route to any trucks is feasible
        Args:
            trucks::List of truck objects
                the list of trucks we want to check if drone route is feasible to
            consec_checks::Boolean
                True if we want to continue checking route to other points, False otherwise
            save_points::Boolean
                True if we want a returned dictionary of truck id to possible feasible points pair
        Returns:
            checker_drone if consec_checks is True, 
            saved_points if save_points is True,
            else 1 if feasible, 0 otherwise
        '''
        if consec_checks and save_points:
            return "In check_truck, consec_checks and save_points cannot be true at the same time. Please change one to false."
        min_turn = 1e3
        target_truck = None
        target_point = None
        saved_points = {}
        for truck in trucks:
            for point in truck.visited_points:
                if point[2] <= self.travel_turn or point[2] > self.travel_turn+self.battery_level:
                    continue
                checker_drone = copy.deepcopy(self) 
                checker_drone.travel_to(Point(point[0], point[1]), diagonal_first=True)
                if checker_drone.visited_points[-1][2] == point[2] and checker_drone.battery_level >= 0:
                    if checker_drone.travel_turn < min_turn:
                        min_turn = checker_drone.travel_turn
                        target_truck = truck 
                        target_point = Point(point[0], point[1])
                    saved_points[target_truck.id] = (point[0], point[1], point[2])
        if not consec_checks and target_point != None:
            if save_points:
                return saved_points
            else:
                return True
        if not consec_checks:
            return False
        elif target_truck == None:
            return False
        else:
            checker_drone = copy.deepcopy(self) 
            checker_drone.travel_to(target_point, diagonal_first=True)
            target_truck.charge_to(checker_drone, checker_drone.battery_capacity)
            target_truck.replenish_drone(checker_drone, checker_drone.item_capacity)
            return checker_drone



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

    # def charge_required(source, dest):
    #     '''Check charge or time required to travel from one point to another
    #         Args:
    #             source::Tuple (x,y)
    #                 (x,y) coordinate of starting point
    #             dest::Tuple (x,y)
    #                 (x,y) coordinate of destination point               
    #         '''
    #     charge_req = 0
    #     source_x, source_y = source
    #     dest_x, dest_y = dest

    #     direct_line = (dest_x == source_x or dest_y == source_y)
        
    #     while not direct_line:
    #         source_x += 1 if source_x < dest_x else -1
    #         source_y += 1 if source_y < dest_y else -1
    #         charge_req += 1       

    #         direct_line = (dest_x== source_x or dest_y == source_y)

    #     direct_diag = (abs(dest_x-source_x) == abs(dest_y-source_y))
    #     while not direct_diag:
    #         if abs(dest_x-source_x) > abs(dest_y-source_y):
    #             source_x += 1 if source_x < dest_x else -1
    #         else:
    #             source_y += 1 if source_y < dest_y else -1     
    #         charge_req += 1          
    #         direct_diag = (abs(dest_x-source_x) == abs(dest_y-source_y))    

    #     return charge_req

    # def drone_cust_check(drone, customer, dvrp, item_or_charge):
    #     '''Check possible charging points for drone after visiting a potential customer
    #         Args:
    #             drone:: Drone object
    #             customer:: Potential customer object
    #             dvrp:: entire current dvrp (if dvrp has charging_route_lst then just need charging route list)  
    #             item_or_change:: binary value, 0 if checking where to replinish item , 1 if checking where to recharge
    #     '''
        
    #     x,y,current_t = customer.x, customer.y, drone.visited_points[-1][2]
    #     low_x = max(0,x - drone.battery_level)
    #     high_x = x+ drone.battery_level

    #     low_y = max(0,y- drone.battery_level)
    #     high_y =y + drone.battery_level   
        
    #     charging_locs = [] #should add a method in dvrp list of all warehouses and truck travelled route then no need keep calc
    #     for w in dvrp.warehouses:
    #         charging_locs.append((w.x,w.y,-1))
    #     for t in dvrp.trucks:
    #         for pt in t.visited_points:
    #             charging_locs.append((pt[0],pt[1],pt[2]))
                
    #     time_to_cust = charge_required((drone.visited_points[-1][0],drone.visited_points[-1][1]),(x,y))

    #     possible_pts = []
    #     for loc in charging_locs:
    #         if loc[2] == -1:
    #             if item_or_charge == 0:
    #                 if (loc[0] >= low_x) & (loc[0]<= high_x) & (loc[1]>= low_y) & (loc[1] <= high_y):
    #                     possible_pts.append(loc)
    #             else: 
    #                 continue
    #         else:
    #             extra_time = charge_required((x,y),(loc[0],loc[1]))
    #             if (loc[0] >= low_x) & (loc[0]<= high_x) & (loc[1]>= low_y) & (loc[1] <= high_y) & ((current_t+time_to_cust+extra_time) == loc[2]):
    #                 possible_pts.append(loc)
    #     return possible_pts 



    def split_route(self):
        ''' Future Initialization for construction heuristic'''

        for c in range(len(self.customers)):
            cust = self.customers[c]
            #print('customer', cust.id)


            # drone checks
            best_drone_time = 1e3
            best_mtd = 0
            
            for d in range(len(self.drones)):
                drone = self.drones[d]
                if cust.demand > drone.item_capacity:
                    break

                drone_time = 1e3
                # 1. drone cust-truck check
                if drone.check_cust(cust, consec_checks=True).check_truck(self.trucks, consec_checks = False):
                    test_drone = drone.check_cust(cust, consec_checks = True)
                    drone_time = test_drone.visited_points[-1][2]
                    mtd = 1
                 #   print('mtd 1 drone time', drone_time)
                    
                # 2. drone- warehouse - cust- truck check
                #if drone.check_wh(self.warehouses , consec_checks = True).check_cust(cust, consec_checks = True).check_truck(self.trucks):
                    
                elif drone.check_wh(self.warehouses, consec_checks = True).check_cust(cust,consec_checks = True).check_truck(self.trucks, consec_checks = False):
                    test_drone = drone.check_wh(self.warehouses, consec_checks = True)
                #    print('warehosue check', test_drone.visited_points)
                    test_drone = test_drone.check_cust(cust, consec_checks = True)
                    drone_time = test_drone.visited_points[-1][2]
                    mtd = 2
                   # print('mtd 2 drone time', drone_time)

                # 3. drone - truck - cust- truck check
                elif drone.check_truck(self.trucks, consec_checks = True).check_cust(cust, consec_checks = True).check_truck(self.trucks, consec_checks = False):
                    test_drone = drone.check_truck(self.trucks, consec_checks = True).check_cust(cust, consec_checks = True)
                    drone_time = test_drone.visited_points[-1][2]
                    mtd = 3
                
                if drone_time < best_drone_time:
                    #print('cust',cust,'mtd',mtd,'drone_time', drone_time,'best_drone_time',best_drone_time,'inventory',drone.items)
                    best_drone_time = drone_time
                    best_check = test_drone
                    best_drone = drone
                    best_mtd=  mtd
                    
            mtd_cnt = 0
            if best_drone_time < 1e3:
                #print('best mtd',best_mtd,'best check', best_check.visited_points)
                for point in best_check.visited_points:
                    if point[2]> best_drone.travel_turn:
                        best_drone.travel_to(Point(point[0],point[1]),diagonal_first = True)
                        if (best_mtd == 2) & (mtd_cnt ==1): #warehouse point
                            best_drone.replenish_inve()
                        mtd_cnt +=1
                        # need to somehow get where wh and truck is
                        
                        # if drone reaches wh , refill inve
                        
                        #if drone reaches truck , wait till truck arrives, refill inve, charge_to
                print('cust', cust.id, 'served by drone', best_drone.id, 'mtd', best_mtd, 'cust.x,cust.y', (cust.x,cust.y), 'drone loc', (best_drone.visited_points[-1][0],best_drone.visited_points[-1][1]))
                #print('drone point visited',best_drone.visited_points)
                best_drone.serve_customer(cust)
                
                continue
            
            # truck checks    
            best_truck_time = 1e3
            for truck in self.trucks:
                
                time = truck.time_to_point(cust)
               # print('truck',truck, 'time', time, 'best time',best_truck_time)
                if time < best_truck_time:
                    best_truck_time = time
                    truck_idx = truck.id
            
                    
            direction = self.trucks[truck_idx].vert_hor(self.customers[c:], cust, self.drones[0])
            self.trucks[truck_idx].travel_to(cust,direction)
            self.trucks[truck_idx].serve_customer(cust)
            print('cust',cust.id, 'served by truck', truck.id)
                           
        
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

        return sum([c.turn_served for c in self.customers])
        #return max([len(t.visited_points) for t in self.trucks] + [len(d.visited_points) for d in self.drones])

