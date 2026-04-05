"""
Parser for AMPL .dat format instances.

Reads problem instances in .dat format with:
- Ships (N): each with arrival time, handling times per berth, cargo type and quantity
- Berths (M): with throughput capacity
- Cargo types (K): with consumption rates and minimum levels
- Initial inventory levels (L)
"""

import math
from model import Ship, Berth, Inventory, Instance


def parse_dat_file(filepath):
    """
    Parse a .dat file and return an Instance object.
    
    Expected .dat format:
    ```
    set N := 1 2 ... n;
    set M := 1 2 ... m;
    set K := 1 2 ... k;
    set L := 1 2 ... l;
    
    param Mares := <berth values>;
    param v := <throughput values per berth>;
    param a := <arrival times per ship>;
    param e := <handling time exponents per berth>;
    param ck := <consumption rates per cargo type>;
    param q := <cargo quantities per ship and type>;
    ```
    
    Args:
        filepath: Path to the .dat file
        
    Returns:
        Instance object with loaded ships, berths, inventories
        
    Raises:
        ValueError: If file format is invalid
        IOError: If file cannot be read
    """
    data = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    current_param = None
    buffer = []
    
    for line in lines:
        line = line.strip()
        
        if not line or line.startswith('#'):
            continue
        
        if line.startswith('set '):
            parse_set(line, data)
        
        elif line.startswith('param '):
            if ':=' in line:
                parts = line.split(':=')
                param_name = parts[0].replace('param', '').strip()
                values_str = parts[1].strip()
                
                if values_str and not values_str.endswith(';'):
                    current_param = param_name
                    buffer = []
                    if values_str:
                        buffer.append(values_str)
                else:
                    parse_param(param_name, [values_str.rstrip(';')], data)
            else:
                current_param = line.replace('param', '').strip().rstrip(':=').strip()
                buffer = []
        
        elif current_param and line.endswith(';'):
            buffer.append(line.rstrip(';'))
            parse_param(current_param, buffer, data)
            current_param = None
            buffer = []
        
        elif current_param:
            buffer.append(line)
    
    instance = build_instance(data)
    return instance


def parse_set(line, data):
    """Parse a set definition line."""
    match = line.replace('set', '').strip().split(':=')
    if len(match) == 2:
        set_name = match[0].strip()
        values_str = match[1].strip().rstrip(';')
        values = values_str.split()
        data[set_name] = [int(v) if v.isdigit() else v for v in values]


def parse_param(param_name, buffer, data):
    """Parse a parameter definition."""
    values_str = ' '.join(buffer).rstrip(';')
    values_str = values_str.replace('(', ' ').replace(')', ' ').replace(',', ' ')
    tokens = values_str.split()
    
    try:
        if param_name in ['Mares', 'v', 'a', 'e', 'ck']:
            data[param_name] = [float(v) for v in tokens if v]
        else:
            data[param_name] = tokens
    except ValueError:
        data[param_name] = [v for v in tokens if v]


def build_instance(data):
    """
    Build Instance object from parsed data.
    
    Args:
        data: Dictionary with parsed parameters
        
    Returns:
        Instance object
    """
    N = len(data.get('N', []))
    M = len(data.get('M', []))
    K = len(data.get('K', []))
    L = len(data.get('L', []))
    
    if N == 0 or M == 0 or K == 0 or L == 0:
        raise ValueError("Invalid instance: missing set sizes")
    
    ships = []
    berths = []
    inventories = []
    
    arrivals = data.get('a', [0] * N)
    v_throughput = data.get('v', [1] * M)
    e_exponents = data.get('e', [1] * M)
    consumption_rates = data.get('ck', [1] * K)
    initial_levels = data.get('Mares', [10] * K)
    
    arrivals = arrivals[:N]
    v_throughput = v_throughput[:M]
    e_exponents = e_exponents[:M]
    consumption_rates = consumption_rates[:K]
    initial_levels = initial_levels[:K]
    
    for i in range(N):
        ship = Ship()
        ship.id = i
        ship.arrival_time = int(arrivals[i]) if i < len(arrivals) else 0
        ship.remaining_arrival_time = ship.arrival_time
        
        ship.handling_times = []
        for b in range(M):
            if i < len(data.get('q', [])):
                cargo_qty = float(data.get('q', [1])[i]) if i < len(data.get('q', [])) else 1.0
                throughput = float(v_throughput[b]) if b < len(v_throughput) else 1.0
                handling_time = math.ceil(cargo_qty / throughput) if throughput > 0 else 1
            else:
                handling_time = 1
            ship.handling_times.append(handling_time)
        
        ship.cargo_type = i % K
        ship.cargo_quantity = float(data.get('q', [1])[i]) if i < len(data.get('q', [])) else 1.0
        
        ships.append(ship)
    
    for b in range(M):
        berth = Berth(remaining_time=0, cargo_type=b % K, throughput=v_throughput[b] if b < len(v_throughput) else 1.0)
        berths.append(berth)
    
    for k in range(K):
        inv = Inventory(
            level=initial_levels[k] if k < len(initial_levels) else 10,
            consumption_rate=consumption_rates[k] if k < len(consumption_rates) else 1,
            minimum_level=0
        )
        inventories.append(inv)
    
    return Instance(ships, berths, inventories)


def parse_dat_advanced(filepath):
    """
    Advanced parser for complex .dat formats with multi-dimensional parameters.
    
    Handles:
    - Multi-indexed parameters like q(i,k)
    - Handling times h(i,l)
    - Proper assignment of cargo types from q matrix
    
    Args:
        filepath: Path to the .dat file
        
    Returns:
        Instance object with loaded ships, berths, inventories
    """
    data = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    import re
    
    set_matches = re.findall(r'set\s+(\w+)\s*:=\s*([^;]+);', content)
    for name, values in set_matches:
        values_list = values.strip().split()
        data[name] = [int(v) if v.isdigit() else v for v in values_list]
    
    param_matches = re.findall(r'param\s+(\w+)\s*:=\s*([^;]+);', content)
    for name, values in param_matches:
        tokens = re.findall(r'\(.*?\)|\S+', values.strip())
        parsed_values = {}
        
        for token in tokens:
            if '(' in token:
                match = re.match(r'\(([^)]+)\)\s*([\d.]+)', token)
                if match:
                    key = tuple(match.group(1).split(','))
                    val = match.group(2)
                    parsed_values[key] = float(val)
            else:
                try:
                    parsed_values[len(parsed_values)] = float(token)
                except:
                    pass
        
        data[name] = parsed_values if parsed_values else [float(v) for v in tokens if v.replace('.', '').isdigit()]
    
    instance = build_instance_advanced(data)
    return instance


def build_instance_advanced(data):
    """
    Build Instance from advanced parsed data with multi-dimensional parameters.
    
    Args:
        data: Dictionary with parsed sets and parameters
        
    Returns:
        Instance object
    """
    N = len(data.get('N', []))
    M = len(data.get('M', []))
    K = len(data.get('K', []))
    L = len(data.get('L', []))
    
    if N == 0 or M == 0 or K == 0 or L == 0:
        raise ValueError("Invalid instance: missing set sizes")
    
    ships = []
    berths = []
    inventories = []
    
    arrivals = data.get('a', {})
    throughput = data.get('v', {})
    cargo_matrix = data.get('q', {})
    consumption_rates = data.get('ck', {})
    initial_levels = data.get('Mares', {})
    
    if isinstance(arrivals, dict):
        arrivals = [arrivals.get(i, 0) for i in range(N)]
    if isinstance(throughput, dict):
        throughput = [throughput.get(b, 1) for b in range(M)]
    if isinstance(consumption_rates, dict):
        consumption_rates = [consumption_rates.get(k, 1) for k in range(K)]
    if isinstance(initial_levels, dict):
        initial_levels = [initial_levels.get(k, 10) for k in range(K)]
    
    for i in range(N):
        ship = Ship()
        ship.id = i
        ship.arrival_time = int(arrivals[i]) if i < len(arrivals) else 0
        ship.remaining_arrival_time = ship.arrival_time
        
        ship.handling_times = []
        total_cargo = 0
        ship_cargo_type = None
        
        if isinstance(cargo_matrix, dict):
            for k in range(K):
                cargo_qty = cargo_matrix.get((i, k), 0)
                if cargo_qty > 0 and ship_cargo_type is None:
                    ship_cargo_type = k
                    total_cargo = cargo_qty
        else:
            total_cargo = cargo_matrix[i] if i < len(cargo_matrix) else 1.0
            ship_cargo_type = i % K
        
        for b in range(M):
            if total_cargo > 0:
                v_b = throughput[b] if b < len(throughput) else 1.0
                handling_time = math.ceil(total_cargo / v_b) if v_b > 0 else 1
            else:
                handling_time = 1
            ship.handling_times.append(handling_time)
        
        ship.cargo_type = ship_cargo_type if ship_cargo_type is not None else (i % K)
        ship.cargo_quantity = total_cargo
        
        ships.append(ship)
    
    for b in range(M):
        berth = Berth(
            remaining_time=0,
            cargo_type=b % K,
            throughput=throughput[b] if b < len(throughput) else 1.0
        )
        berths.append(berth)
    
    for k in range(K):
        inv = Inventory(
            level=initial_levels[k] if k < len(initial_levels) else 10,
            consumption_rate=consumption_rates[k] if k < len(consumption_rates) else 1,
            minimum_level=0
        )
        inventories.append(inv)
    
    return Instance(ships, berths, inventories)
