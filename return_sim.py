import random
import simpy
import numpy as np


# Total_packages = 1000
Return_percent = 10
MAX_TRUCKS = 10
Truck_capacity = 200

Truck_wait_time = 24*60

Package_wait_time = 5

Travel_time = 12*60

Sim_time = 30*24*60

class Truck:
    def __init__(self, env, Truck_capacity, source, condition, destination):
        self.Trucks_capacity = simpy.Container(env, init=0,capacity=Truck_capacity)
        self.env = env
        self.source = source
        self.condition = condition
        self.destination = destination
        # self.locs = locs

    def wait(self):
        yield self.env.timeout(Truck_wait_time)
        print(f"Not waiting anymore {self.source.name}")
        if not self.condition.triggered:
            self.condition.succeed()
        # self.go()

    def go(self, travel_time):
        yield self.condition
        self.source.current_truck[self.destination] = None
        print(f"Truck from {self.source.name} to {self.destination.name} is departing at {self.env.now:.2f} with {self.Trucks_capacity._level}")
        yield self.env.timeout(Travel_time)
        print(f"Truck from {self.source.name} to {self.destination.name} has reached at {self.env.now:.2f} with {self.Trucks_capacity._level}")
        yield self.destination.Trucks.put(1)
        # if self.source.name =="0":
        #     yield self.locs[1].Trucks.put(1)
        # else:
        #     yield self.locs[0].Trucks.put(1)

class Location:
    def __init__(self, env, Trucks, name):
        self.env = env
        self.Trucks = simpy.Container(env,init=Trucks, capacity=Trucks)
        self.current_truck = {}
        self.name = name
        self.locs = [self]
        self.travel_time ={}

    def process_package(self, package, destination):
        if destination in self.current_truck and self.current_truck[destination] is not None:
            # yield self.env.timeout(0.1)
            yield self.current_truck[destination].Trucks_capacity.put(1)
            # print(self.name , self.current_truck.Trucks_capacity._level)
        else:
            # print(self.Trucks._level)

            yield self.Trucks.get(1)
            print(f"Truck assigned at source {self.name} for destination {destination.name} at time {self.env.now:.2f}")
            self.current_truck[destination] = Truck(env = self.env,
                                                    Truck_capacity= Truck_capacity,
                                                    source=self,
                                                    condition=simpy.Event(self.env),
                                                    destination= destination)
            self.env.process(self.current_truck[destination].go(travel_time=self.travel_time[destination]))
            self.env.process(self.current_truck[destination].wait())
            yield self.current_truck[destination].Trucks_capacity.put(1)

            # print(self.current_truck.Trucks_capacity._level)

        if self.current_truck[destination].Trucks_capacity._level == self.current_truck[destination].Trucks_capacity.capacity:
            print(f"bhar gaya at source {self.name}")
            if not self.current_truck[destination].condition.triggered:
                yield self.current_truck[destination].condition.succeed()
            # .succeed()

    def connect(self, other, travel_time):
        self.locs.append(other)
        self.travel_time[other] = travel_time
        other.locs.append(self)
        other.travel_time[self] = travel_time

    def random_start(self, destination):
        for i in range(0, random.randrange(80, 150)):
            yield self.env.timeout(0.1)
            self.env.process(self.process_package(i, destination))


class customer_emulator:
    def __init__(self, source, destination, package_wait_time, package_wait_time_std,env):
        self.env = env
        self.source = source
        self.destination = destination
        self.package_wait_time = package_wait_time
        self.package_wait_time_std = package_wait_time_std

    def emulate(self):
        while True:
            random_wait = max(0.01, np.random.normal(self.package_wait_time, self.package_wait_time_std))
            yield self.env.timeout(random_wait)
            self.env.process(self.source.process_package(0,self.destination))

def setup():
    env = simpy.Environment()

    locs = []
    for i in range(4):
        locs.append(Location(env, i+1,str(i)))

    for i in range(4):
        for j in range(i+1,4):
            locs[i].connect(locs[j], travel_time=Travel_time)



    for i in range(4):
        for j in range(4):
            if i != j:
                locs[i].random_start(locs[j])
                env.process(customer_emulator(source =locs[i],
                                  env=env,
                                  destination=locs[j],
                                  package_wait_time=Package_wait_time,
                                  package_wait_time_std=3).emulate())
    env.run(until=Sim_time)


setup()

