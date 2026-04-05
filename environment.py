"""
Port environment for reinforcement learning.

This module provides a PyEnvironment interface for the port berth allocation
problem without generator dependencies. The environment receives instances directly.
"""

import numpy as np
import math
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import constants
from simulator import Simulator
import copy


class PortEnvironment(py_environment.PyEnvironment):
    """Port environment for RL experiments."""
    
    def __init__(self, parameters, instance, mode=constants.TRAINING):
        """
        Initialize the environment.
        
        Args:
            parameters: Dictionary with environment parameters:
                - 'max-ships': Maximum ships to moor
                - 'lookahead-size': Lookahead window size
                - 'max-handling-time': Maximum handling time
                - 'max-arrival-time': Maximum arrival time
                - 'max-inventory-level': Maximum inventory level
                - 'min-consumption-rate': Minimum consumption rate
                - 'max-consumption-rate': Maximum consumption rate
                - 'penalty': Invalid action penalty
            instance: Instance object with ships, berths, inventories
            mode: TRAINING or TESTING mode
        """
        super().__init__()
        
        self.mode = mode
        self.parameters = parameters
        self.instance = instance
        
        self.prev_ttw = 0
        self.current_ttw = 0
        self.delta_ttw = 0
        self.current_reward = 0

        self.num_invalid_actions = 0
        self.num_collapsed_inventory = 0

        self.ships = copy.deepcopy(self.instance.ships)
        self.berths = copy.deepcopy(self.instance.berths)
        self.inventories = copy.deepcopy(self.instance.inventories)
        
        self.num_ships = len(self.ships)
        self.num_berths = len(self.berths)
        self.num_inventories = len(self.inventories)

        self.state = None

        self.num_queues = 2
        self.num_selected_queue = np.zeros(self.num_queues)
        self.num_queue_default_selected = 0

        self.num_actions = (self.num_queues + 1) ** self.num_berths

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=self.num_actions - 1, name='action')

        observation_ranges = self.num_berths * 4 + self.num_inventories * 2 + self.parameters['lookahead-size'] * (2 + 2 * self.num_berths)

        minimum = [0.] * observation_ranges
        maximum = [1.] * observation_ranges

        self._observation_spec = {
            'observation': array_spec.BoundedArraySpec(
                (observation_ranges,), dtype=np.float32, minimum=minimum, maximum=maximum, name='observation'),
            'mask': array_spec.ArraySpec(shape=(self.num_actions,), dtype=bool, name='mask')
        }

        self.simulator = Simulator(self.ships, self.berths, self.inventories, self.parameters)
        self.state = self.simulator.get_observation()
        self.state_normalized = self.observation_to_state()
        
        self.current_criticality = self.get_criticality()
        self.min_idle_time = self.parameters['max-handling-time']

    def action_spec(self):
        """Return action spec."""
        return self._action_spec

    def observation_spec(self):
        """Return observation spec."""
        return self._observation_spec

    def _obs(self, obs):
        """Create observation with mask."""
        masking = [True] * self.num_actions
        masking[0] = False

        criticalities = self.get_criticality()
        neg_criticality = False
        for c in criticalities:
            if c < 0:
                neg_criticality = True
                break

        for action_num in range(self.num_actions):
            queue = [None] * self.num_berths
            actual_number = action_num
            for b in range(self.num_berths - 1):
                queue[b] = actual_number // ((self.num_queues + 1) ** (self.num_berths - b - 1))
                actual_number = actual_number % ((self.num_queues + 1) ** (self.num_berths - b - 1))

            queue[self.num_berths - 1] = actual_number

            for b in range(self.num_berths):
                if queue[b] == 2:
                    if neg_criticality:
                        masking[action_num] = False
                
                if obs[b] * queue[b] != 0 or obs[b] + queue[b] == 0:
                    masking[action_num] = False
                    break

        return {'observation': obs, 'mask': np.array(masking, dtype=bool)}

    def _step(self, action_input_number):
        """Execute one step in the environment."""
        handling_time = np.zeros(self.num_berths)
        action = self.get_action_from_input_number(action_input_number)

        mooring_list = []
        for berth, ship in enumerate(action):
            if ship >= 0:
                mooring_list.append([berth, ship])

        idle_time = 0
        ships = []
        for b, berth in enumerate(self.berths):
            if not berth.is_available():
                ships.append(berth.ship)
                handling_time[b] = ships[b].handling_times[b]
            else:
                ships.append(self.simulator.get_ship_by_id(action[b]))
                handling_time[b] = ships[b].handling_times[b]
                
                idle_time += ships[b].remaining_arrival_time
                if ships[b].remaining_arrival_time < self.min_idle_time:
                    self.min_idle_time = ships[b].remaining_arrival_time

        self.state, reward, done, info = self.simulator.update_state(mooring_list)

        if info in [constants.DUPLICATED, constants.NO_AVAILABITY, constants.NO_ACTIONS]:
            self.num_invalid_actions += 1
        elif info == constants.COLLAPSED_INVENTORY:
            self.num_collapsed_inventory += 1

        state_normalized = self.observation_to_state()
        self.update_delta_ttw(self.simulator.ttw)

        partial_reward = 0
        for b in range(self.num_berths):
            partial_reward += min(self.delta_ttw, max(self.current_ttw - ships[b].arrival_time, 0)) / max(handling_time[b], 1)

        if reward > 0:
            new_criticality = self.get_criticality()
            transformed_reward = max(1 - self.calculate_energy_function(self.current_criticality, new_criticality) - 0.1 * (idle_time - self.min_idle_time), 0)
            self.current_criticality = new_criticality
        else:
            transformed_reward = reward / (len(self.simulator.moored_ships) + 1)

        self.current_reward = transformed_reward

        if done:
            return ts.termination(self._obs(np.array(state_normalized, dtype=np.float32)), reward=transformed_reward)
        elif self.num_collapsed_inventory == 0:
            return ts.transition(self._obs(np.array(state_normalized, dtype=np.float32)), reward=transformed_reward)
        else:
            return ts.truncation(self._obs(np.array(state_normalized, dtype=np.float32)), reward=transformed_reward)

    def _reset(self):
        """Reset the environment."""
        self.num_invalid_actions = 0
        self.num_collapsed_inventory = 0
        self.num_selected_queue = np.zeros(self.num_queues)
        self.num_queue_default_selected = 0

        self.ships = copy.deepcopy(self.instance.ships)
        self.berths = copy.deepcopy(self.instance.berths)
        self.inventories = copy.deepcopy(self.instance.inventories)

        self.simulator = Simulator(self.ships, self.berths, self.inventories, self.parameters)

        self.prev_ttw = 0
        self.update_delta_ttw(self.simulator.ttw)

        self.state = self.simulator.get_observation()
        state_normalized = self.observation_to_state()
        
        self.current_criticality = self.get_criticality()

        return ts.restart(self._obs(np.array(state_normalized, dtype=np.float32)))

    def observation_to_state(self):
        """Convert raw observation to normalized state."""
        state = []

        state += self.normalize(self.state[4], 0, self.parameters['max-handling-time'] + self.parameters['max-arrival-time'])
        state += self.normalize(self.state[5], 0, self.num_inventories - 1)
        state += self.normalize(self.state[6], 0, self.parameters['max-arrival-time'])
        state += self.normalize(self.state[7], 0, 10)
        
        consumption_rates = [inv.consumption_rate for inv in self.instance.inventories]
        state += self.normalize(consumption_rates, self.parameters['min-consumption-rate'], self.parameters['max-consumption-rate'])

        state += self.normalize(self.get_criticality(), 0, self.parameters['max-inventory-level'])
        state += self.normalize(self.get_inventory_remaining_times(), 0, int(self.parameters['max-inventory-level'] / max(self.parameters['min-consumption-rate'], 1)))

        state += self.normalize(self.state[0], 0, self.parameters['max-arrival-time'])

        handling_times = self.state[1]
        for h in handling_times:
            state += self.normalize(h, 1, self.parameters['max-handling-time'])
           
        state += self.normalize(self.state[2], 0, self.num_inventories - 1)

        return state

    def get_criticality(self):
        """Calculate criticality for each inventory."""
        inventory_criticalities = [None] * self.num_inventories
        maximum_remaining_eta = max([s.remaining_arrival_time for s in self.simulator.lookahead] + [0])
        
        for k, inv in enumerate(self.inventories):
            if inv.consumption_rate == 0:
                inventory_criticalities[k] = 0
            else:
                inventory_criticalities[k] = (inv.level / inv.consumption_rate) - maximum_remaining_eta
                for ship in self.simulator.lookahead:
                    if ship.cargo_type == k:
                        if inv.consumption_rate > 0:
                            inventory_criticalities[k] = max((inv.level / inv.consumption_rate) - ship.remaining_arrival_time, inventory_criticalities[k])
                        break
            
        return inventory_criticalities

    def get_inventory_remaining_times(self):
        """Calculate remaining time for each inventory."""
        remaining_times = [None] * self.num_inventories
        for i, inv in enumerate(self.inventories):
            if inv.consumption_rate > 0:
                remaining_times[i] = math.ceil(inv.level / inv.consumption_rate)
            else:
                remaining_times[i] = 0
            
        return remaining_times

    def get_action_from_input_number(self, input_number):
        """Convert action number to ship selection for each berth."""
        action_input_array = []

        queue = [None] * self.num_berths
        actual_number = input_number
        for b in range(self.num_berths - 1):
            queue[b] = actual_number // ((self.num_queues + 1) ** (self.num_berths - b - 1))
            actual_number = actual_number % ((self.num_queues + 1) ** (self.num_berths - b - 1))

        queue[self.num_berths - 1] = actual_number
        
        completion_queue = [None] * self.num_berths
        for b in range(self.num_berths):
            completion_queue[b] = copy.deepcopy(self.simulator.lookahead)
            completion_queue[b].sort(key=lambda s: max(s.arrival_time, self.simulator.ttw) + s.handling_times[b] - 1)

        lookahead = copy.deepcopy(self.simulator.lookahead)
        ships_cargo_type = [[] for _ in range(self.num_inventories)]
        
        for s in lookahead:
            ships_cargo_type[int(s.cargo_type)].append(s.id)

        lookahead_ids = [s.id for s in self.simulator.lookahead]
        minimum_criticality_list = []

        for b in range(self.num_berths):
            ship = -1
            if queue[b] == 1:
                completion_queue_ids = [s.id for s in completion_queue[b]]
                ship = self.get_first_new_ship(completion_queue_ids, action_input_array)
                if ship == -1:
                    ship = self.get_first_new_ship(lookahead_ids, action_input_array)
            elif queue[b] == 2:
                criticalities = self.get_criticality()
                filtered_criticalities = [v for v in criticalities if v not in minimum_criticality_list]
    
                if filtered_criticalities:
                    min_criticality = min(filtered_criticalities)
                    minimum_criticality_list.append(min_criticality)
                else:
                    min_criticality = None
                
                if min_criticality is not None:
                    k = criticalities.index(min_criticality)
                    ship = self.get_first_new_ship(ships_cargo_type[k], action_input_array)
                    if ship == -1:
                        ship = self.get_first_new_ship(lookahead_ids, action_input_array)
                        self.num_queue_default_selected += 1

            action_input_array.append(ship)

        for b in range(self.num_berths):
            if queue[b] > 0:
                self.num_selected_queue[queue[b] - 1] += 1

        return action_input_array

    def get_first_new_ship(self, policy_queue, selected_ships):
        """Get first ship from queue not yet selected."""
        for ship in policy_queue:
            if ship not in selected_ships:
                return ship
        return -1

    def normalize(self, x, min_value, max_value):
        """Normalize values to [0, 1] range."""
        if min_value == max_value:
            return x
        else:
            return [min((x_i - min_value) / (max_value - min_value), 1) for x_i in x]

    def update_delta_ttw(self, current_ttw):
        """Update time difference."""
        self.prev_ttw = self.current_ttw
        self.current_ttw = current_ttw
        self.delta_ttw = self.current_ttw - self.prev_ttw
        return self.delta_ttw

    def calculate_energy_function(self, state, next_state):
        """Calculate energy function for reward shaping."""
        energy = 0
        energy_total = 0
        alpha = 0.8
        beta = 0.25
        
        for i in range(self.num_inventories):
            derror = (next_state[i] - state[i])

            if next_state[i] < 1:
                error = next_state[i] / 10
            elif next_state[i] < 3:
                error = next_state[i] / 5
            elif next_state[i] < 5:
                error = next_state[i] / 2
            else:
                error = next_state[i]

            energy_total += 1 / (max(error, 0) + 1)

            if derror < 0:
                energy += 1 / (max(error, 0) + 1)

        energy_index = min(alpha * energy + beta * energy_total, 0.99)
        return energy_index

    def render(self):
        """Render environment state."""
        self.simulator.render()
        print(f"Invalid actions: {self.num_invalid_actions}, Collapsed inventories: {self.num_collapsed_inventory}")
        print(f"Selected queues: {self.num_selected_queue}")
