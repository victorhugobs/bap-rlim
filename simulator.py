"""
Simulator for the port berth allocation problem with inventory management.

This module contains the core simulation logic for mooring ships at berths
and managing cargo inventory consumption.
"""

import numpy as np
import copy
import constants
from model import Ship, Berth, Inventory


class Simulator:
    """Core simulator for ship-berth allocation with inventory management."""
    
    def __init__(self, ships, berths, inventories, parameters):
        """
        Initialize the simulator.
        
        Args:
            ships: List of Ship objects
            berths: List of Berth objects
            inventories: List of Inventory objects
            parameters: Dictionary with simulation parameters:
                - 'penalty': Reward penalty for invalid actions
                - 'max-ships': Maximum number of ships to moor
                - 'lookahead-size': Number of ships to consider in lookahead
        """
        self.ships = ships
        self.ships.sort(key=lambda s: s.arrival_time)
        
        self.unmoored_ships = copy.deepcopy(self.ships)
        self.moored_ships = []
        self.berths = berths
        self.berth_moors = [[] for _ in range(len(self.berths))]
        self.num_artificial_ships = 0
        self.inventories = inventories
        self.parameters = parameters
        self.lookahead = []
        self.unloading_rates = np.zeros(len(self.inventories))
        self.ttw = 0
        self.total_reward = 0
        self.service_time = 0
        self.num_collapsed_inventory = 0

        self.ordering_ships()
        self.create_lookahead_list()

    def update_state(self, mooring_list):
        """
        Update simulator state given a mooring action.
        
        Args:
            mooring_list: List of (berth_id, ship_id) tuples
            
        Returns:
            Tuple (observation, reward, done, info)
        """
        done = False
        info = {}
        reward = 0
        accumulated_service_time = 0

        if len(mooring_list) == 0:
            reward = self.parameters['penalty']
            self.total_reward += reward
            print("NO_ACTIONS")
            info = constants.NO_ACTIONS
            return self.get_observation(), reward, done, info

        for s in mooring_list:
            if not self.berths[s[0]].is_available():
                reward = self.parameters['penalty']
                self.total_reward += reward
                print('NO-AVAILABILITY: ', s, s[0], self.berths[s[0]].remaining_time)
                info = constants.NO_AVAILABITY
                return self.get_observation(), reward, done, info

        ships = [s[1] for s in mooring_list]
        for s in ships:
            if ships.count(s) > 1:
                reward = self.parameters['penalty']
                self.total_reward += reward
                print("DUPLICATED")
                info = constants.DUPLICATED
                return self.get_observation(), reward, done, info

        self.ordering_ships()
        self.create_lookahead_list()
        
        for s in mooring_list:
            berth = s[0]
            ship_id = s[1]
            ship = self.get_ship_by_id(ship_id)
            
            self.moor_ship(ship, berth)

            if ship_id >= self.parameters['max-ships']:
                self.num_artificial_ships += 1

            accumulated_service_time += self.calculate_reward(ship, berth)
            
            self.berths[berth].remaining_time = ship.handling_times[berth] + max(ship.arrival_time - self.ttw, 0)
            self.berths[berth].cargo_type = ship.cargo_type

            self.ordering_ships()
            self.create_lookahead_list()

        reward = accumulated_service_time
        self.service_time += accumulated_service_time

        is_collapsed_inventory = self.next_state()

        if is_collapsed_inventory:
            reward = self.parameters['penalty']
            info = constants.COLLAPSED_INVENTORY

        self.total_reward += reward
        self.state = self.get_observation()

        if len(self.moored_ships) >= self.parameters['max-ships'] and not is_collapsed_inventory:
            done = True

        return self.state, reward, done, info

    def next_state(self):
        """Advance time until next berth becomes available."""
        is_collapsed_inventory = False
        
        while not self.is_any_available_berth():
            self.update_remaining_arrival_times()
            self.update_remaining_berth_times()
            self.update_inventory_levels()
            if self.is_any_collapsed_level():
                is_collapsed_inventory = True 

            self.ttw += 1

        return is_collapsed_inventory

    def get_observation(self):
        """Get current observation state."""
        state = []

        remaining_arrival_times = []
        handling_times = []
        ship_cargo_types = []
        cargo_quantities = []
        contributions = []

        for s in self.lookahead:
            remaining_arrival_times.append(s.remaining_arrival_time)
            handling_times.append(s.handling_times)
            ship_cargo_types.append(s.cargo_type)
            cargo_quantities.append(s.cargo_quantity)
            
            contributions.append(
                [s.cargo_quantity - self.inventories[s.cargo_type].consumption_rate * ht
                for ht in s.handling_times]
            )
            
        state.append(remaining_arrival_times)
        state.append(handling_times)
        state.append(ship_cargo_types)
        state.append(cargo_quantities)

        remaining_berth_times = []
        berth_cargo_types = []
        unloading_rates = []
        idle_berth_times = []

        for i, b in enumerate(self.berths):
            remaining_berth_times.append(b.remaining_time)
            berth_cargo_types.append(b.cargo_type)
            
            if b.remaining_time == 0:
                idle_berth_times.append(0)
                unloading_rates.append(0)
            else:
                idle_berth_times.append(max(self.ttw - b.ship.arrival_time, 0))
                unloading_rates.append(b.ship.cargo_quantity / b.ship.handling_times[i])
        
        state.append(remaining_berth_times)
        state.append(berth_cargo_types)
        state.append(idle_berth_times)
        state.append(unloading_rates)

        inventory_levels = []
        for i in self.inventories:
            inventory_levels.append(i.level)

        state.append(inventory_levels)
        state.append(contributions)

        return state

    def is_any_collapsed_level(self):
        """Check if any inventory has collapsed."""
        for inv in self.inventories:
            if inv.is_collapsed_level():
                return True
        return False

    def calculate_reward(self, ship, berth):
        """Calculate service time reward."""
        arrival_time = ship.arrival_time
        handling_time = ship.handling_times[berth]
        service_time = handling_time + max(self.ttw - arrival_time, 0)
        return service_time

    def update_remaining_arrival_times(self):
        """Decrease remaining arrival times for all ships."""
        for ship in self.ships:
            ship.remaining_arrival_time = max(ship.remaining_arrival_time - 1, 0)

        for ship in self.unmoored_ships:
            ship.remaining_arrival_time = max(ship.remaining_arrival_time - 1, 0)

    def update_remaining_berth_times(self):
        """Decrease remaining completion times for all berths."""
        for berth in self.berths:
            berth.remaining_time = max(berth.remaining_time - 1, 0)

    def update_unloading_rates(self):
        """Calculate current unloading rates from all berths."""
        self.unloading_rates = np.zeros(len(self.inventories))
        for i, berth in enumerate(self.berths):
            if berth.ship is not None:
                if self.is_operation_initialized(berth):
                    ship = berth.ship
                    unloading_rate = ship.cargo_quantity / ship.handling_times[i]
                    self.unloading_rates[ship.cargo_type] += unloading_rate

    def update_inventory_levels(self):
        """Update inventory levels based on unloading and consumption."""
        self.update_unloading_rates()

        for i, inv in enumerate(self.inventories):
            inv.level = max(inv.level + self.unloading_rates[i] - inv.consumption_rate, 0)
            if inv.level <= 0:
                self.num_collapsed_inventory += 1

    def ordering_ships(self):
        """Sort unmoored ships by arrival time."""
        self.unmoored_ships.sort(key=lambda s: s.arrival_time)

    def create_lookahead_list(self):
        """Create look-ahead list of next ships to moor."""
        self.lookahead = self.unmoored_ships[:self.parameters['lookahead-size']]

    def is_available_berth(self, berth):
        """Check if specific berth is available."""
        return self.berths[berth].is_available()

    def is_any_available_berth(self):
        """Check if any berth is available."""
        for berth in self.berths:
            if berth.is_available():
                return True
        return False

    def moor_ship(self, ship, b):
        """Moor a ship at a berth."""
        berth = self.berths[b]
        berth.ship = ship
        for s in self.unmoored_ships:
            if s.id == ship.id:
                self.unmoored_ships.remove(s)
                self.moored_ships.append(s)
                self.berth_moors[b].append([s.id, self.ttw, [round(inventory.level, 1) for inventory in self.inventories]])
                break

    def get_ship_by_id(self, id):
        """Get ship object by ID."""
        for s in self.ships:
            if s.id == id:
                return s
        return None

    def get_ship_by_lookahead_index(self, index):
        """Get ship from lookahead list by index."""
        return self.lookahead[index]

    def is_operation_initialized(self, berth):
        """Check if ship operation has started at berth."""
        ship = berth.ship
        if ship.arrival_time > self.ttw:
            return False
        else:
            return True

    def render(self):
        """Print current simulator state."""
        print("=== SIMULATOR STATE ===")
        print("Unmoored ships:")
        for u in self.unmoored_ships:
            print(f"  Ship {u.id}: Arrival={u.arrival_time}, Cargo Type={u.cargo_type}, Quantity={u.cargo_quantity}")
        
        print("Moored ships:")
        for m in self.moored_ships:
            print(f"  Ship {m.id}: Arrival={m.arrival_time}, Cargo Type={m.cargo_type}")
        
        print("Berth status:")
        for i, b in enumerate(self.berths):
            print(f"  Berth {i}: Remaining time={b.remaining_time}")
        
        print("Inventories:")
        for i, inv in enumerate(self.inventories):
            print(f"  Inventory {i}: Level={inv.level}, Consumption={inv.consumption_rate}, Min={inv.minimum_level}")
        
        print(f"Mooring records: {self.berth_moors}")
        print(f"TTW: {self.ttw}")
        print(f"Artificial ships: {self.num_artificial_ships}")
        print(f"Total reward: {self.total_reward}")
        print(f"Total service time: {self.service_time}")
