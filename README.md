# Port Berth Allocation with Inventory Management

A reutilizable simulation framework for the port berth allocation problem (BAP) with inventory control, designed for reinforcement learning research and reproducibility.

## Features

- **Modular Design**: Decoupled simulator, environment, and instance definitions
- **Light Dependencies**: Only NumPy, TensorFlow, and TF-Agents
- **RL-Ready**: Compatible with TF-Agents PyEnvironment framework
- **Easy Extensibility**: Simple to override and customize for your research
- **Parser Support**: Read problem instances from `.dat` files

## Project Structure

```
├── model.py                    # Data classes (Ship, Berth, Inventory, Instance)
├── simulator.py                # Core simulation engine
├── environment.py              # RL environment (TF-Agents compatible)
├── parser.py                   # .dat file parser
├── run_example.py              # Usage examples
├── constants.py                # Constants and enums
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Create and Run a Simple Episode

```python
from model import Ship, Berth, Inventory, Instance
from environment import PortEnvironment
import constants

# Create problem instance
instance = Instance(ships=[...], berths=[...], inventories=[...])

# Create environment
params = {
    'penalty': -100,
    'max-ships': 10,
    'lookahead-size': 5,
    'max-handling-time': 10,
    'max-arrival-time': 50,
    'max-inventory-level': 150,
    'min-consumption-rate': 2,
    'max-consumption-rate': 8,
}
env = PortEnvironment(params, instance, mode=constants.TRAINING)

# Run episode
time_step = env.reset()
while True:
    action = env.action_spec().sample()
    time_step = env.step(action)
    if time_step.step_type == 2:  # Termination
        break
```

### 2. Run the Example

```bash
python run_example.py
```

This runs two demonstrations:
- **Random Policy**: Takes random valid actions
- **Greedy Policy**: Always selects earliest ship

### 3. Parse a .dat Instance

```python
from parser import parse_dat_file

instance = parse_dat_file('instance.dat')
```

Expected `.dat` format:
```
set N := 1 2 ... n;
set M := 1 2 ... m;
set K := 1 2 ... k;
set L := 1 2 ... l;

param Mares := <initial_inventory_levels>;
param v := <berth_throughputs>;
param a := <ship_arrival_times>;
param e := <berth_exponents>;
param ck := <consumption_rates>;
param q := <cargo_quantities>;
```

## Components Overview

### `model.py` - Data Classes

- **Ship**: Represents a vessel with arrival time, cargo, and handling times
- **Berth**: Port berth with availability status and throughput
- **Inventory**: Cargo inventory with consumption rate and minimum level
- **Instance**: Container for a complete problem instance

### `simulator.py` - Simulation Engine

Core logic for:
- Ship mooring at berths
- Inventory consumption and updates
- Time advancement
- Reward calculation
- State observation

**Key Methods**:
- `update_state(mooring_list)`: Execute one simulator step
- `get_observation()`: Get current state
- `reset()`: Initialize for new episode (via environment)

### `environment.py` - RL Environment

TensorFlow Agents compatible environment providing:
- Action/observation specifications
- Step and reset methods
- Action masking (invalid action prevention)
- Policy queue selection (completion time, criticality)

**Usage with RL Agents**:
```python
from tf_agents.agents.dqn import dqn_agent

env = PortEnvironment(params, instance)
# Use with any TF-Agents agent
q_net = ...
agent = dqn_agent.DqnAgent(...)
```

### `parser.py` - Instance Parser

- `parse_dat_file()`: Basic parser for `.dat` format
- `parse_dat_advanced()`: Advanced parser for multi-dimensional parameters
- Automatic handling time calculation: `h(i,l) = ceil(q_i / v_l)`

## Problem Definition

**Port Berth Allocation as Reinforcement Learning with Inventory Management (BAP-RLIM)**

**Inputs**:
- **Ships (N)**: Each with arrival time, cargo type, quantity, handling times per berth
- **Berths (M)**: Each with throughput capacity
- **Inventories (K)**: Cargo types with consumption rates and minimum levels

**Objective**:
- Minimize total service time (waiting + handling)
- Constrain: Avoid inventory collapse

**State Space**:
- Remaining arrival times (lookahead)
- Handling times per berth
- Berth occupancy
- Current inventory levels
- Criticality metrics

**Action Space**:
- Select a ship for each available berth
- Queue strategies: earliness, completion time, criticality

## Extending the Framework

### Add a Custom Agent

```python
from environment import PortEnvironment
from tf_agents.agents import DqnAgent

env = PortEnvironment(params, instance)
agent = DqnAgent(
    time_step_spec=env.time_step_spec(),
    action_spec=env.action_spec(),
    ...
)
```

### Custom Reward Function

Override `calculate_energy_function()` in `PortEnvironment`:

```python
def calculate_energy_function(self, state, next_state):
    # Your reward shaping logic
    return reward
```

### Custom Policy

Override `get_action_from_input_number()` to implement different queuing policies.

## Parameters

**Required Parameters Dictionary**:
- `penalty`: Negative reward for invalid actions (default: -100)
- `max-ships`: Maximum ships to serve per episode
- `lookahead-size`: Ships to consider in next steps
- `max-handling-time`: Maximum service time (for normalization)
- `max-arrival-time`: Maximum arrival time (for normalization)
- `max-inventory-level`: Maximum inventory level (for normalization)
- `min-consumption-rate`: Minimum consumption (for normalization)
- `max-consumption-rate`: Maximum consumption (for normalization)

## Constants (`constants.py`)

```python
# Action info codes
NO_ACTIONS = 0
NO_AVAILABITY = 1
DUPLICATED = 2
COLLAPSED_INVENTORY = 4

# Mode codes
TRAINING = 0
TESTING_INSTANCE = 1
TESTING_BUFFER_INSTANCES = 4
TESTING_RANDOM_INSTANCES = 2
GENERATING_INSTANCES = 3
```

## Typical Workflow

```
1. Create Instance (programmatically or parse .dat)
   ↓
2. Define Parameters
   ↓
3. Initialize Environment
   ↓
4. Run RL Agent Training/Testing
   ↓
5. Evaluate Results (rewards, moored ships, service time)
```

## Reproducibility

- All randomness can be controlled via NumPy seeds
- Deterministic instance parsing
- Parameter configuration recorded in code

```python
import numpy as np
np.random.seed(42)

instance = create_deterministic_instance()
env = PortEnvironment(params, instance)
```

## License

Academic use. See original project for attribution.

## Citation

If you use this framework, please cite the original port allocation literature and this adaptation.

## Contact

For questions or issues with this version, refer to the main project repository.
