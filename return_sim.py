import random
import simpy
import numpy as np
import csv
from collections import defaultdict
from queue import PriorityQueue
import os

# Constants
new_truck = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
Return_percent = 15 # redundant
Warehouse_capacity =1000 #assuming infinite capacity for normal deliveries for now
MAX_TRUCKS = 10
Truck_capacity = 200
Truck_wait_time = 24 * 60  # minutes
demand_delay = 10  # minutes
Travel_time = 12 * 60  # minutes
Sim_time = 30 * 24 * 60  # minutes
Max_wait_until_return = 80 #minutes
returns_prevented = 0
cost_per_package = 1
total_packages = 0
packages_returned = 0
packages_intended_for_return = 0
number_of_packages = 0
orders = 0
no_of_truck_trips = 0
avg_util = 0
package_discarded = 0

#Product Name : {Product ID, Product Type, Brand
product_details = {
    "P1": [0,0, 0],
    "P2": [1,2, 1],
    "P3": [2,1, 0],
    "P4": [3,1, 2],
    "P5": [4,1, 1],
    "P6": [5,2, 2]
}

Probability_stats = {
    "Product_ID" : [0.89, 0.23, 0.67, 0.77, 0.12, 0.56],
    "Product_Type": [0.56, 0.91, 0.87],
    "Brand": [0.34, 0.88, 0.11],
    "Location":[0,0.4,0.9]
}

#weights for each parameter
weights = {
    "Product_ID":0.5,
    "Product_Type":0.4,
    "Brand":0.1
}

#calculating probabilities
probabilities = []
for product in product_details:
    product_id = product_details[product][0]
    product_type = product_details[product][1]
    brand = product_details[product][2]
    product_prob = (weights["Product_ID"]*Probability_stats["Product_ID"][product_id]
                             +weights["Product_Type"]*Probability_stats["Product_Type"][product_type]
                             +weights["Brand"]*Probability_stats["Brand"][brand])
    probabilities.append(product_prob)
sum_prob = sum(probabilities)
probabilities = [p / sum_prob for p in probabilities]


products = ["P1","P2","P3","P4","P5","P6"]
# Global variables for cost tracking
total_cost = 0

class Package:
    def __init__(self, env, condition, source, destination,product, return_status=False):
        self.env = env
        self.condition = condition
        self.source = source
        self.destination = destination
        self.return_status = 0 if np.random.random()>(Return_percent/100) else 1
        self.truck = None  # Reference to the truck carrying the package
        self.initial_delivery_cost = 0  # Cost of delivering this package initially
        self.product = product
        #print(self.product)
        global packages_intended_for_return, number_of_packages
        number_of_packages += 1
        # if self.return_status:


    def wait(self):
        yield self.env.timeout(Max_wait_until_return)
        # self.destination.warehouse.inventory[self.product].put((self.env.now, self))
        if not self.condition.triggered:
            #print("package return")
            self.condition.succeed()
            global packages_returned
            # yield self.condition
            # if self.condition.ok:
            packages_returned += 1
            self.dispatch_product_to_source()

    def dispatch_product_to_source(self):
        self.source, self.destination = self.destination, self.source
        self.return_status = False
        self.env.process(self.source.process_package(self))



class Warehouse:
    def __init__(self, env, location):
        self.env = env
        self.location = location
        self.inventory = defaultdict(lambda: PriorityQueue())  # keeps track of returned packages ONLY
        # self.warehouse_storage = simpy.Container(env, init=0, capacity=Warehouse_capacity)
        # self.Warehouse.warehouse_storage(self, env, location=location)

    def check_for_package(self, package_id):
        # print(self.inventory[package_id].empty())
        if not self.inventory[package_id].empty():
            return True
        else:
            return False

    def load_package_to_warehouse(self, package_id, package):

        self.inventory[package_id].put((self.env.now, package))

    def dispatch_returned_product_for_new_order(self, package_id):
        global returns_prevented, package_discarded
        # print("h")
        _, package = self.inventory[package_id].get()

        while not self.inventory[package_id].empty() and package.condition.triggered:
            _, package = self.inventory[package_id].get()
            package_discarded+=1
        if package.condition.triggered:
            package_discarded += 1
            return False
        package.condition.succeed()
        returns_prevented += 1
        return True

class Truck:
    def __init__(self, env, truck_capacity, source, condition, destination, return_trip=False):
        self.env = env
        self.truck_capacity = simpy.Container(env, init=0, capacity=truck_capacity)
        self.source = source
        self.condition = condition
        self.destination = destination
        self.packages = []  # List to hold packages being transported
        self.return_trip = return_trip  # Flag to indicate if this truck is making a return trip

    def wait(self):
        yield self.env.timeout(Truck_wait_time)
        if not self.condition.triggered:
            self.condition.succeed()
            #print(f"Not waiting anymore from {self.source} to {self.destination} at {self.env.now:.2f}")

    def go(self):
        yield self.condition
        global total_cost, total_packages, no_of_truck_trips
        # self.load_packages()
        self.source.current_truck[self.destination] = None
        num_packages = self.truck_capacity._level
        #print(f"Truck from {self.source.name} to {self.destination.name} is departing at {self.env.now:.2f} with {num_packages} packages")
        yield self.env.timeout(Travel_time)
        #print(f"Truck from {self.source.name} to {self.destination.name} has reached at {self.env.now:.2f}")

        global total_cost, cost_per_package, packages_returned, avg_util

        # Cost of initial delivery
        total_cost += Travel_time * 0.01
        #print(total_packages)
        total_packages += num_packages
        avg_util += num_packages/Truck_capacity
        #print(num_packages/Truck_capacity)

        no_of_truck_trips += 1
        # Return truck to source location if needed

        self.env.process(self.unload_truck())
        yield self.destination.trucks.put(1)
         #determine the order

    def unload_truck(self):
        global packages_intended_for_return
        for package in self.packages:
            if package.return_status:
                packages_intended_for_return += 1
                package.destination.warehouse.load_package_to_warehouse(package.product, package)
                #print(f"start wait {self.env.now:.2f}")
                # self.env.process(package.dispatch_product_to_source())
                self.env.process(package.wait())
                yield self.env.timeout(0.01)
                # print(f"end wait {self.env.now:.2f}")
                #package.destination.warehouse.return_package(package.product)

                package.source, package.destination = package.destination, package.source



class Location:
    def __init__(self, env, trucks, name):
        self.env = env
        self.trucks = simpy.Container(env, init=trucks, capacity=MAX_TRUCKS)
        self.current_truck = defaultdict(lambda: None)
        self.name = name
        self.locs = [self]
        self.travel_time = {}
        self.packages = []  # List of packages waiting to be shipped
        self.warehouse = None

    def assign_warehouse(self, warehouse):
        self.warehouse = warehouse

    def process_package(self, package):
        global new_truck
        """Process a package and assign it to a truck if available."""
        # if package.return_status:
            # package.destination.warehouse.load_package_to_warehouse(package.product)
            # print(f"start wait {self.env.now:.2f}")
            # package.wait()
            # package.dispatch_product_to_source()
            # print(f"end wait {self.env.now:.2f}")
            # #package.destination.warehouse.return_package(package.product)
            # package.source, package.destination = package.destination, package.source

        if package.destination in self.current_truck and self.current_truck[package.destination] is not None:
            truck = self.current_truck[package.destination]
            yield truck.truck_capacity.put(1)  # Load package if truck has capacity
            truck.packages.append(package)
            package.truck = truck  # Reference the truck carrying the package
        else:
            yield self.trucks.get(1)
            #print(f"{new_truck} Truck assigned at source {self.name} for destination {package.destination.name} at time {self.env.now:.2f}")

            if self.current_truck[package.destination] is None:
                new_truck[int(package.source.name)][int(package.destination.name)] += 1
                self.current_truck[package.destination] = Truck(env=self.env,
                                                            truck_capacity=Truck_capacity,
                                                            source=self,
                                                            condition=simpy.Event(self.env),
                                                            destination=package.destination)

            else:
                yield self.trucks.put(1)
            truck = self.current_truck[package.destination]
            self.env.process(truck.go())
            self.env.process(truck.wait())
            truck.packages.append(package)
            package.truck = truck
            self.current_truck[package.destination].truck_capacity.put(1)
            #print(package.source.name)

        if self.current_truck[package.destination].truck_capacity._level == self.current_truck[package.destination].truck_capacity.capacity:
            #print(f"bhar gaya at source {self.name} for {package.destination.name} at {self.env.now:.2f}")
            if not self.current_truck[package.destination].condition.triggered:
                yield self.current_truck[package.destination].condition.succeed()

    def connect(self, other, travel_time):
        self.locs.append(other)
        self.travel_time[other] = travel_time
        other.locs.append(self)
        other.travel_time[self] = travel_time

    def random_start(self, destination):
        for i in range(random.randrange(80, 150)):
            yield self.env.timeout(0.1)

            package = Package(i, self, destination, return_status=(random.random() < Return_percent / 100))
            self.env.process(self.process_package(package))

class CustomerEmulator:
    def __init__(self, source, destination, demand_delay, demand_delay_std, env):
        self.env = env
        self.source = source
        self.destination = destination
        self.demand_delay = demand_delay
        self.demand_delay_std = demand_delay_std

    def emulate(self, env):
        while True:
            random_wait = max(0.01, np.random.normal(self.demand_delay, self.demand_delay_std))
            yield self.env.timeout(random_wait)
            global orders
            orders += 1
            product_name = np.random.choice(products, p=probabilities)
            # product_name = 0
            # print(self.destination.warehouse.check_for_package(product_name[0]))
            #check if the ordered product is already at the destinaiton due to a prior return
            if self.destination.warehouse.check_for_package(product_name):
                # print('hi')
                if self.destination.warehouse.dispatch_returned_product_for_new_order(product_name):
                    continue
                #print(self.destination.warehouse.inventory)
                pass

            # else:
            package = Package(
                self.env,
                condition=simpy.Event(self.env),
                source = self.source,
                destination = self.destination,
                product=product_name
            )
            #yield Warehouse(env, location = package.destination).warehouse_storage


            self.env.process(self.source.process_package(package))

def setup():
    global total_cost
    total_cost = 0

    env = simpy.Environment()

    locs = []
    for i in range(4):
        locs.append(Location(env, 9, str(i)))

    for i in range(4):
        locs[i].assign_warehouse(Warehouse(env, location=locs[i]))
        for j in range(i + 1, 4):
            locs[i].connect(locs[j], travel_time=Travel_time)

    for i in range(4):
        for j in range(4):
            if i != j:
                locs[i].random_start(locs[j])
                env.process(CustomerEmulator(source=locs[i],
                                             destination=locs[j],
                                             demand_delay=demand_delay,
                                             demand_delay_std=1,
                                             env=env).emulate(env))

    env.run(until=Sim_time)

    # Print final results
    print(f"Total cost of transportation: {total_cost}")
    print(f"Total number of packages: {total_packages}, {number_of_packages}, {orders}")
    #print(f"Total cost of returns: {total_return_cost}")
    #print(f"Total cost of delivering returned packages (excluding return trips): {total_delivery_cost}")
    #print(f"Total cost of return trips: {total_return_trip_cost}")
    print(f"Duplicates prevented:{returns_prevented}")
    print(f"Packages Returned: {packages_returned}, {package_discarded}")
    print(f"Packages intended to be returned: {packages_intended_for_return}")
    print(f"Number of Truck Trips: {no_of_truck_trips}")
    print(f"Average Utilization: {avg_util/no_of_truck_trips}, {new_truck}")

    # Prepare metrics for CSV
    metrics = [
        total_cost,
        total_packages,
        orders,
        number_of_packages,
        returns_prevented,
        packages_returned,
        package_discarded,
        packages_intended_for_return,
        no_of_truck_trips,
        avg_util / no_of_truck_trips if no_of_truck_trips > 0 else 0,
    ]

    # Define CSV file path
    csv_file = 'simulation_metrics.csv'

    # Check if file exists and write header if not
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header only if the file is newly created
            header = [
                "Total cost of transportation",
                "Total number of packages",
                "Total number of orders",
                "Total number of orders loaded to trucks"
                "Duplicates prevented",
                "Packages Returned",
                "Packages discarded",
                "Packages intended to be returned",
                "Number of Truck Trips",
                "Average Utilization",
            ]
            writer.writerow(header)
        # Write the metrics as a new row
        writer.writerow(metrics)

setup()


def run_multiple_simulations(max_wait_times, num_cycles):
    results = {time: {'total_cost': 0, 'total_orders': 0, 'total_packages_returned': 0, 'total_orders_loaded_to_truck':0,
                      'total_prevented': 0, 'total_discarded': 0, 'total_intended_for_return': 0,
                      'total_truck_trips': 0, 'total_avg_util': 0} for time in max_wait_times}

    for wait_time in max_wait_times:
        for _ in range(num_cycles):
            global Max_wait_until_return
            Max_wait_until_return = wait_time
            # Reset global variables
            global total_cost, total_packages, number_of_packages, orders
            global returns_prevented, packages_returned, package_discarded, packages_intended_for_return
            global no_of_truck_trips, avg_util, new_truck
            total_cost = 0
            total_packages = 0
            number_of_packages = 0
            orders = 0
            returns_prevented = 0
            packages_returned = 0
            package_discarded = 0
            packages_intended_for_return = 0
            no_of_truck_trips = 0
            avg_util = 0
            new_truck = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

            # Run the simulation
            setup()  # Calls your existing setup function

            # Collect metrics
            avg_util_per_trip = avg_util / no_of_truck_trips if no_of_truck_trips > 0 else 0
            cost_per_order = total_cost / orders if orders > 0 else 0
            cost_wasted_on_returns = 2 * cost_per_package * packages_returned

            results[wait_time]['total_cost'] += total_cost
            results[wait_time]['total_orders'] += orders
            results[wait_time]['total_packages_returned'] += packages_returned
            results[wait_time]['total_orders_loaded_to_truck'] += number_of_packages
            results[wait_time]['total_prevented'] += returns_prevented
            results[wait_time]['total_discarded'] += package_discarded
            results[wait_time]['total_intended_for_return'] += packages_intended_for_return
            results[wait_time]['total_truck_trips'] += no_of_truck_trips
            results[wait_time]['total_avg_util'] += avg_util_per_trip

    # Calculate averages
    avg_results = {time: {key: value / num_cycles for key, value in metrics.items()} for time, metrics in
                   results.items()}

    return avg_results

import matplotlib.pyplot as plt

def plot_results(avg_results):
    global cost_per_package
    wait_times = list(avg_results.keys())
    cost_per_package = [metrics['total_cost']/(metrics['total_orders_loaded_to_truck']) for metrics in avg_results.values()]
    print(cost_per_package)
    for i in range(len(cost_per_package)):
        avg_cost_wasted = [2 * cost_per_package[i] * metrics['total_discarded'] for metrics in avg_results.values()]
    avg_utilizations = [metrics['total_avg_util']*100 for metrics in avg_results.values()]
    for metrics in avg_results.values():
        print(metrics)
    percent_prevented = [metrics['total_prevented'] / metrics['total_intended_for_return'] * 100 if metrics['total_intended_for_return'] > 0 else 0 for metrics in avg_results.values()]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Max Wait Until Return (minutes)')
    ax1.set_ylabel('Average Cost Wasted', color=color)
    ax1.plot(wait_times, avg_cost_wasted, color=color, label='Average Cost Wasted')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('% of Returns Shipped as New Orders', color=color)
    ax2.plot(wait_times, avg_utilizations, color='tab:orange', label='Average Utilization')
    ax2.plot(wait_times, percent_prevented, color=color, linestyle='--', label='% Prevented from Return')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Average Cost Wasted and Utilization vs Max Wait Until Return')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

def run_simulations_with_return_rates(return_rates, num_cycles):
    results = {
        rate: {'total_cost': 0, 'total_orders': 0, 'total_packages_returned': 0, 'total_orders_loaded_to_truck': 0,
               'total_prevented': 0, 'total_discarded': 0, 'total_intended_for_return': 0,
               'total_truck_trips': 0, 'total_avg_util': 0} for rate in return_rates}

    for rate in return_rates:
        for _ in range(num_cycles):
            global Return_percent
            Return_percent = rate

            # Reset global variables
            global total_cost, total_packages, number_of_packages, orders
            global returns_prevented, packages_returned, package_discarded, packages_intended_for_return
            global no_of_truck_trips, avg_util, new_truck
            total_cost = 0
            total_packages = 0
            number_of_packages = 0
            orders = 0
            returns_prevented = 0
            packages_returned = 0
            package_discarded = 0
            packages_intended_for_return = 0
            no_of_truck_trips = 0
            avg_util = 0
            new_truck = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

            # Run the simulation
            setup()  # Calls your existing setup function

            # Collect metrics
            avg_util_per_trip = avg_util / no_of_truck_trips if no_of_truck_trips > 0 else 0
            cost_per_order = total_cost / orders if orders > 0 else 0
            cost_wasted_on_returns = 2 * cost_per_package * packages_returned

            results[rate]['total_cost'] += total_cost
            results[rate]['total_orders'] += orders
            results[rate]['total_packages_returned'] += packages_returned
            results[rate]['total_orders_loaded_to_truck'] += number_of_packages
            results[rate]['total_prevented'] += returns_prevented
            results[rate]['total_discarded'] += package_discarded
            results[rate]['total_intended_for_return'] += packages_intended_for_return
            results[rate]['total_truck_trips'] += no_of_truck_trips
            results[rate]['total_avg_util'] += avg_util_per_trip

    # Calculate averages
    avg_results = {rate: {key: value / num_cycles for key, value in metrics.items()} for rate, metrics in
                   results.items()}

    return avg_results

def plot_results_with_return_rates(avg_results):
    return_rates = list(avg_results.keys())
    cost_per_package = [metrics['total_cost'] / (metrics['total_orders_loaded_to_truck']) for metrics in avg_results.values()]
    avg_cost_wasted = [2 * cost_per_package[i] * metrics['total_discarded'] for i, metrics in enumerate(avg_results.values())]
    avg_utilizations = [metrics['total_avg_util'] * 100 for metrics in avg_results.values()]
    percent_prevented = [metrics['total_prevented'] / metrics['total_intended_for_return'] * 100 if metrics['total_intended_for_return'] > 0 else 0 for metrics in avg_results.values()]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Return Rate (%)')
    ax1.set_ylabel('Average Cost Wasted', color=color)
    ax1.plot(return_rates, avg_cost_wasted, color=color, label='Average Cost Wasted')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('% of Returns Shipped as New Orders', color=color)
    ax2.plot(return_rates, avg_utilizations, color='tab:orange', label='Average Utilization')
    ax2.plot(return_rates, percent_prevented, color=color, linestyle='--', label='% Prevented from Return')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Average Cost Wasted and Utilization vs Return Rate')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    MAX_WAIT_TIMES = [0, 50, 60, 70, 80, 90]
    NUM_CYCLES = 1

    avg_results = run_multiple_simulations(MAX_WAIT_TIMES, NUM_CYCLES)

    # Write results to CSV
    csv_file = 'simulation_metrics_multiple_cycles.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            header = [
                "Max Wait Time",
                "Average Total Cost",
                "Average Total Orders",
                "Average Total Orders Loaded on Trucks"
                "Average Total Packages Returned",
                "Average Duplicates Prevented",
                "Average Packages Discarded",
                "Average Packages Intended for Return",
                "Average Number of Truck Trips",
                "Average Utilization"
            ]
            writer.writerow(header)
        for wait_time, metrics in avg_results.items():
            writer.writerow([
                wait_time,
                metrics['total_cost'],
                metrics['total_orders'],
                metrics['total_orders_loaded_to_truck'],
                metrics['total_packages_returned'],
                metrics['total_prevented'],
                metrics['total_discarded'],
                metrics['total_intended_for_return'],
                metrics['total_truck_trips'],
                metrics['total_avg_util']
            ])

    # Plot orders placed vs orders shipped
    packages_returned = [metrics['total_discarded'] for metrics in avg_results.values()]
    total_returns_intended = [metrics['total_intended_for_return'] for metrics in avg_results.values()]
    wait_times = list(avg_results.keys())

    plt.figure(figsize=(10, 5))
    plt.plot(wait_times, packages_returned, label='Actual Packages Returned', marker='o')
    plt.plot(wait_times, total_returns_intended, label='Packages Intended for Return', marker='x')
    plt.xlabel('Wait Time')
    plt.ylabel('Returns')
    plt.title('Returns Filed vs Actual Packages Returned to the Merchant')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_results(avg_results)

    demand_delay = 5
    RETURN_RATES = [5, 10, 15, 20, 25, 30]
    NUM_CYCLES = 1

    avg_results = run_simulations_with_return_rates(RETURN_RATES, NUM_CYCLES)

    # Write results to CSV
    csv_file = 'simulation_metrics_return_rates.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            header = [
                "Return Rate (%)",
                "Average Total Cost",
                "Average Total Orders",
                "Average Total Orders Loaded on Trucks",
                "Average Total Packages Returned",
                "Average Duplicates Prevented",
                "Average Packages Discarded",
                "Average Packages Intended for Return",
                "Average Number of Truck Trips",
                "Average Utilization"
            ]
            writer.writerow(header)
        for rate, metrics in avg_results.items():
            writer.writerow([
                rate,
                metrics['total_cost'],
                metrics['total_orders'],
                metrics['total_orders_loaded_to_truck'],
                metrics['total_packages_returned'],
                metrics['total_prevented'],
                metrics['total_discarded'],
                metrics['total_intended_for_return'],
                metrics['total_truck_trips'],
                metrics['total_avg_util']
            ])

    plt.figure(figsize=(10, 5))
    packages_returned = [metrics['total_discarded'] for metrics in avg_results.values()]
    total_returns_intended = [metrics['total_intended_for_return'] for metrics in avg_results.values()]
    rate = list(avg_results.keys())
    print(rate)
    plt.plot(rate, packages_returned, label='Actual Packages Returned', marker='o')
    plt.plot(rate, total_returns_intended, label='Packages Intended for Return', marker='x')
    plt.xlabel('Wait Time')
    plt.ylabel('Returns')
    plt.title('Returns Filed vs Actual Packages Returned to the Merchant')
    plt.legend()
    plt.grid(True)
    plt.show()