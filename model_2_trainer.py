import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from tkinter import messagebox
from collections import deque
from model_2_knapsack_env import KnapsackEnv
from model_2_model import TransformerKnapsackModel
from utils import load_data, calc_approx_ratio
from visual import *

# Hyperparameters
BATCH_SIZE = 512
GAMMA = 0.99
LR = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.999
TARGET_UPDATE_INTERVAL = 100
NUM_EPISODES = 3000
MEMORY_SIZE = 50000
HUBER_DELTA = 2.0

# Training loop
def train_transformer_model(data_type, num_items, visual=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_name = f"training_data_{data_type.lower()}_{num_items}.pkl"
    folder_path = "train_data"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        training_data = load_data(file_path)
    else:
        raise FileNotFoundError(f"Training data file '{file_path}' not found.")
    
    # Prepare data
    values, weights, capacities, solutions, optimal_values = prepare_data(training_data)

    # Initialize environment and model
    env = KnapsackEnv(values, weights, capacities)

    model = TransformerKnapsackModel(input_dim=6, hidden_dim=96, num_layers=2, num_heads=2).to(device)
    target_model = TransformerKnapsackModel(input_dim=6, hidden_dim=96, num_layers=2, num_heads=4).to(device)
    target_model.load_state_dict(model.state_dict())  # Sync with main model initially
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Replay buffer
    replay_buffer = deque(maxlen=MEMORY_SIZE)

    # Define epsilon for exploration
    epsilon = EPSILON_START

    # Visualization setup
    if visual:
        visualizer = KnapsackVisualizer(knapsack_plot=False, reward_plot=True)
    else:
        visualizer = None

    rewards = []
    optimal_rewards = []

    for episode in range(NUM_EPISODES):
        model.train() # Set model to training mode
        state = env.reset()

        total_reward = 0
        selected_items = [0] * int(num_items) # Initialize selected items tracking for this episode
        instance_index = env.get_current_index()
        
        for t in range(int(num_items)):
            # Select and take action
            action = select_action(state, epsilon, model, device)

            # Update selected items based on action (1 = select, 0 = skip)
            selected_items[t] = action

            # Step in environment
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Store experience in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            
            # Train only if enough experience is available
            if len(replay_buffer) >= BATCH_SIZE:
                train_on_batch(model, target_model, optimizer, replay_buffer, device, BATCH_SIZE, GAMMA, HUBER_DELTA)
            
            if done:
                break

        # Decay epsilon
        epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon)

        # Update target network periodically
        if episode % TARGET_UPDATE_INTERVAL == 0:
            target_model.load_state_dict(model.state_dict())


        rewards.append(total_reward) # Record total reward of the current episode
        optimal_rewards.append(optimal_values[instance_index]) # Record optimal reward of the current episode

        # Calculate the  reward approximation ratio for this episode
        approx_ratios = calc_reward_approx(rewards, optimal_rewards)

        # Visualize model selections and approximation ratio after every episode
        if visualizer and visualizer.knapsack_plot:
            visualizer.plot_knapsack(
                torch.tensor(values[instance_index]),   
                torch.tensor(weights[instance_index]),  
                selected_items,             
                capacity=capacities[instance_index][0],
                optimal_solution=solutions[instance_index]
            )

        visualizer.update_reward_plot(approx_ratios) # Update reward plot

        # Log progress
        print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
    
    # Save the trained model
    torch.save(model, f"trained_model_2_{data_type}_{num_items}.pth")
    print("Model saved as 'trained_model_2.pth")

    # Exit visualization
    # Show a pop-up dialog when training is complete
    response = messagebox.showinfo("Training Complete","Training has completed successfully.")

    if response == "ok":
        visualizer.close_knapsack_plot()
    
# Helper functions
def prepare_data(data):
    """Extract values, weights and capacities from a data file"""

    values = []
    weights = []
    capacities = []
    solutions = []
    optimal_values = []

    for items, capacity, solution, objective in data:
        values.append([item[0] for item in items]) # Extract values
        weights.append([item[1] for item in items]) # Extract weights
        capacities.append([capacity] * len(items)) # Repeat capacity for each item
        solutions.append(solution)
        optimal_values.append(objective)

    return values, weights, capacities, solutions, optimal_values

# Action selection function with epsilon-greedy strategy
def select_action(state, epsilon, model, device):
    """Selects action using epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.choice([0, 1])
    else:
        # Ensure each component in state is a scalar value
        try:
            state_tensor = torch.tensor([
                state["capacity"][0] if isinstance(state["capacity"], list) else state["capacity"],         # Should be a single value
                state["bag_weight"],       # Should be a single value
                state["bag_value"],        # Should be a single value
                state["item_weight"],      # Should be a single value
                state["item_value"],       # Should be a single value
                state["remaining_steps"]   # Should be a single value
            ], dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension
        except TypeError as e:
            print("Error in constructing state tensor:", e)
            raise

        with torch.no_grad():
            return model(state_tensor).argmax().item()

def train_on_batch(model, target_model, optimizer, replay_buffer, device, batch_size, gamma, huber_delta):
    """Trains model on a batch sampled from replay buffer."""
    # Experience replay sampling
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Prepare states and next_states as tensors by extracting the values from each dictionary
    states = torch.tensor([
        [
            state["capacity"][0] if isinstance(state["capacity"], list) else state["capacity"],
            state["bag_weight"],
            state["bag_value"],
            state["item_weight"],
            state["item_value"],
            state["remaining_steps"]
        ] for state in states
    ], dtype=torch.float32, device=device)

    next_states = torch.tensor([
        [
            next_state["capacity"][0] if isinstance(next_state["capacity"], list) else next_state["capacity"],
            next_state["bag_weight"],
            next_state["bag_value"],
            next_state["item_weight"],
            next_state["item_value"],
            next_state["remaining_steps"]
        ] for next_state in next_states
    ], dtype=torch.float32, device=device)

    # Convert actions, rewards, and dones to tensors
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    # Compute Q-values for current states
    q_values = model(states)

    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute Q-values for next states with target model
    with torch.no_grad():
        max_next_q_values = target_model(next_states).max(1)[0]
        # Bellman Equation - maximum expected reward in future instances
        target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

    # Compute loss and backpropagate
    loss = nn.SmoothL1Loss(beta=huber_delta)(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Test code

# data_type = "UC"  
# num_items = 10   

# train_transformer_model(data_type, num_items)