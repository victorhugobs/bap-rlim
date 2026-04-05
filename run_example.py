"""
Example demonstrating environment usage.

This example shows how to:
1. Create a problem instance
2. Initialize the environment
3. Run a simple episode with random actions
4. Display results
"""

import numpy as np
from model import Ship, Berth, Inventory, Instance
from environment import PortEnvironment
from simulator import Simulator
import constants


def create_simple_instance(num_ships=10, num_berths=2, num_inventories=4):
    """
    Create a simple problem instance programmatically.
    
    Args:
        num_ships: Number of ships
        num_berths: Number of berths
        num_inventories: Number of cargo types
        
    Returns:
        Instance object
    """
    ships = []
    for i in range(num_ships):
        ship = Ship(
            id=i,
            arrival_time=i * 5,
            handling_times=[5, 6],
            cargo_type=i % num_inventories,
            cargo_quantity=20 + i * 2
        )
        ships.append(ship)
    
    berths = []
    for b in range(num_berths):
        berth = Berth(remaining_time=0, cargo_type=0, throughput=4.0)
        berths.append(berth)
    
    inventories = []
    for k in range(num_inventories):
        inv = Inventory(level=100, consumption_rate=5, minimum_level=20)
        inventories.append(inv)
    
    return Instance(ships, berths, inventories)


def default_parameters(num_berths=2, num_inventories=4):
    """Get default parameters for the environment."""
    return {
        'penalty': -100,
        'max-ships': 10,
        'lookahead-size': 3,
        'max-handling-time': 10,
        'max-arrival-time': 50,
        'max-inventory-level': 150,
        'min-consumption-rate': 2,
        'max-consumption-rate': 8,
        'num-berths': num_berths,
        'num-inventories': num_inventories,
    }


def run_random_episode():
    """Run a simple episode with random actions."""
    print("=" * 60)
    print("PORT ENVIRONMENT EXAMPLE")
    print("=" * 60)
    
    params = default_parameters(num_berths=2, num_inventories=4)
    instance = create_simple_instance(num_ships=10, num_berths=2, num_inventories=4)
    
    print("\n[1] Created problem instance:")
    instance.render()
    
    print("\n[2] Initializing environment...")
    env = PortEnvironment(params, instance, mode=constants.TRAINING)
    print(f"   - Action space: {env.num_actions} possible actions")
    print(f"   - Observation space: {env.observation_spec()['observation'].shape}")
    
    print("\n[3] Running random episode...")
    time_step = env.reset()
    episode_reward = 0
    step_count = 0
    max_steps = 50
    
    while step_count < max_steps:
        valid_actions = np.where(time_step.observation['mask'])[0]
        
        if len(valid_actions) == 0:
            print(f"   Step {step_count}: No valid actions available")
            break
        
        action = np.random.choice(valid_actions)
        
        time_step = env.step(action)
        episode_reward += float(time_step.reward)
        step_count += 1
        
        if step_count % 10 == 0 or time_step.step_type == 2:
            print(f"   Step {step_count}: Reward={time_step.reward:.2f}, Cumulative={episode_reward:.2f}")
        
        if time_step.step_type == 2:
            print(f"   Episode finished at step {step_count}")
            break
    
    print("\n[4] Episode Results:")
    print(f"   - Total steps: {step_count}")
    print(f"   - Total reward: {episode_reward:.2f}")
    print(f"   - Average reward per step: {episode_reward / max(step_count, 1):.2f}")
    print(f"   - Invalid actions: {env.num_invalid_actions}")
    print(f"   - Collapsed inventories: {env.num_collapsed_inventory}")
    
    print("\n[5] Final Simulator State:")
    env.simulator.render()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


def run_greedy_episode():
    """Run an episode using a greedy policy (moor earliest ship first)."""
    print("\n" + "=" * 60)
    print("GREEDY POLICY EXAMPLE")
    print("=" * 60)
    
    params = default_parameters(num_berths=2, num_inventories=4)
    instance = create_simple_instance(num_ships=15, num_berths=2, num_inventories=4)
    
    print("\n[1] Initialized environment with greedy policy")
    env = PortEnvironment(params, instance, mode=constants.TRAINING)
    
    print("[2] Running greedy episode (always select action 1)...")
    time_step = env.reset()
    episode_reward = 0
    step_count = 0
    max_steps = 100
    
    while step_count < max_steps:
        valid_actions = np.where(time_step.observation['mask'])[0]
        
        if len(valid_actions) == 0:
            print(f"   No valid actions at step {step_count}")
            break
        
        action = valid_actions[0]
        
        time_step = env.step(action)
        episode_reward += float(time_step.reward)
        step_count += 1
        
        if time_step.step_type == 2:
            break
    
    print(f"\n[3] Greedy Policy Results:")
    print(f"   - Total steps: {step_count}")
    print(f"   - Total reward: {episode_reward:.2f}")
    print(f"   - Moored ships: {len(env.simulator.moored_ships)}")
    print(f"   - Total service time: {env.simulator.service_time}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_random_episode()
    run_greedy_episode()
