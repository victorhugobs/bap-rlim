"""
Data model classes for the port berth allocation problem.

Classes:
    Ship: Represents a vessel with arrival time, cargo, and handling times per berth
    Berth: Represents a port berth with availability status
    Inventory: Represents a cargo inventory with consumption rate
    Instance: Container for a complete problem instance
"""

class Ship:
    """Represents a ship with cargo and handling requirements."""
    
    def __init__(self, id=None, arrival_time=None, handling_times=None, cargo_type=None, cargo_quantity=0):
        """
        Initialize a Ship.
        
        Args:
            id: Ship identifier
            arrival_time: Time when ship arrives at port
            handling_times: List of handling times for each berth
            cargo_type: Type of cargo (0-indexed inventory type)
            cargo_quantity: Quantity of cargo
        """
        self.id = id
        self.arrival_time = arrival_time
        self.remaining_arrival_time = self.arrival_time
        self.handling_times = handling_times if handling_times is not None else []
        self.cargo_type = cargo_type
        self.cargo_quantity = cargo_quantity
        self.has_been_moored = False


class Berth:
    """Represents a berth in the port."""
    
    def __init__(self, remaining_time=0, cargo_type=None, ship=None, throughput=None):
        """
        Initialize a Berth.
        
        Args:
            remaining_time: Time remaining for current operation
            cargo_type: Type of cargo currently handled
            ship: Ship currently moored at this berth
            throughput: Berth throughput (vessels/time)
        """
        self.remaining_time = remaining_time
        self.cargo_type = cargo_type
        self.ship = ship
        self.throughput = throughput

    def is_available(self):
        """Check if berth is available for mooring."""
        return self.remaining_time <= 0


class Inventory:
    """Represents a cargo inventory."""
    
    def __init__(self, level=0, consumption_rate=0, minimum_level=0):
        """
        Initialize an Inventory.
        
        Args:
            level: Current inventory level
            consumption_rate: Rate of consumption per time unit
            minimum_level: Minimum acceptable level
        """
        self.level = level
        self.consumption_rate = consumption_rate
        self.minimum_level = minimum_level

    def is_collapsed_level(self):
        """Check if inventory has fallen below minimum level."""
        return self.level <= self.minimum_level


class Instance:
    """Container for a complete problem instance."""
    
    def __init__(self, ships, berths, inventories):
        """
        Initialize an Instance.
        
        Args:
            ships: List of Ship objects
            berths: List of Berth objects
            inventories: List of Inventory objects
        """
        self.ships = ships
        self.berths = berths
        self.inventories = inventories

    def render(self):
        """Print instance information."""
        print('=== INSTANCE ===')
        print(f'Number of ships: {len(self.ships)}')
        print(f'Number of berths: {len(self.berths)}')
        print(f'Number of inventories: {len(self.inventories)}')
        print()
        
        print('Ships:')
        for s in self.ships:
            print(f'  ID: {s.id}, Arrival: {s.arrival_time}, Cargo Type: {s.cargo_type}, Quantity: {s.cargo_quantity}')
        print()
        
        print('Berths:')
        for idx, b in enumerate(self.berths):
            print(f'  Berth {idx}: Available={b.is_available()}')
        print()
        
        print('Inventories:')
        for idx, i in enumerate(self.inventories):
            print(f'  Inventory {idx}: Level={i.level}, Consumption={i.consumption_rate}, Min={i.minimum_level}')
