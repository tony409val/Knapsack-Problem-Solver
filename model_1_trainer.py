import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from model_1_model import NeuralKnapsackSolver
from utils import *
from visual import *
from tkinter import messagebox


# Helper function to prepare data for Pytorch Dataloader
def prepare_data(data):
    # Extract values, weights, capacities, and solutions from the data
    values = []
    weights = []
    capacities = []
    solutions = []

    for items, capacity, solution, objective in data:
        values.append([item[0] for item in items]) # Extract values
        weights.append([item[1] for item in items]) # Extract weights
        capacities.append([capacity] * len(items)) # Repeat capacity for each item
        solutions.append(solution) # Optimal solution (0 or 1 for each item)

    # Convert to Pytorch tensors
    values = torch.tensor(values, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)
    capacities = torch.tensor(capacities, dtype=torch.float32).unsqueeze(-1) # Add extra dimension
    solutions = torch.tensor(solutions, dtype=torch.float32)

    # Concatenate values, weights, and capacities to form the input tensor
    inputs = torch.cat([values.unsqueeze(-1), weights.unsqueeze(-1), capacities], dim=-1)  # Shape: (batch_size, num_items, 3)


    return inputs, solutions

# Training Function
def train_knapsack_solver(data_type, num_items, visual, num_epochs=100, batch_size=100, learning_rate=0.004, max_wait=5):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate instances
    #training_data = generate_and_solve_instances(instance_type, num_instances, num_items, value_weight_range, H)

    # Load Training Data
    file_name = f"training_data_{data_type.lower()}_{num_items}.pkl"
    folder_path = "train_data"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        training_data = load_data(file_path)
    else:
        raise FileNotFoundError(f"Training data file '{file_path}' not found.")


    inputs, solutions = prepare_data(training_data)
    dataset = TensorDataset(inputs, solutions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Define model, loss function, and optimizer
    input_dim = 3 # (value, weight, capacity)
    hidden_dim = 32
    model = NeuralKnapsackSolver(input_dim, hidden_dim).to(device)

    criterion = nn.BCELoss() # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # best_approx_ratio = float('inf') # Initialize best approximation ratio
    # wait = 0 # Counter for early stopping

    # Initialize visual plot
    if visual:
        visualizer = KnapsackVisualizer(knapsack_plot=False, approx_plot=True)
    else:
        visualizer = None
    
    avg_ratios = []

    # Training Loop
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        total_approx_ratio = 0.0
        num_instances = 0

        for inputs, targets in dataloader:

            # Move inputs and targets to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Extract values, weights, capacities from the input tensor
            values = inputs[:, :, 0] # Extract values
            weights = inputs[:, :, 1] # Extract weights
            capacities = inputs[:, :, 2] # Extract capacities
           
            # Forward pass
            outputs = model(values, weights, capacities)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # Gradient Clipping
            optimizer.step()

            # Track running loss
            running_loss += loss.item()

            # Calculate approximation ratio
            approx_ratio = calc_approx_ratio(outputs, targets, values)
            total_approx_ratio += approx_ratio
            num_instances +=1

            # Plot knapsack selections every `plot_interval` epochs
            if visualizer and visualizer.knapsack_plot:
                if  num_instances == 1:  # Plot only once per epoch on first batch
                    model_solution = outputs[0].round().tolist()  # First item in batch for plotting
                    optimal_solution = targets[0].tolist()  # First item in batch (ground truth)

                    visualizer.plot_knapsack(
                        values[0],
                        weights[0],
                        model_solution,
                        capacities[0][0].item(),
                        optimal_solution
                    )

        
        # Calculate average approximation ratio
        avg_approx_ratio = total_approx_ratio / num_instances

        avg_ratios.append(avg_approx_ratio)

        # Plot average approximation ratio progress
        if visualizer:
            if len(avg_ratios) > 2:
                visualizer.update_approx_plot(avg_ratios)

        # Print loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        print(f"Epoch {epoch + 1}/{num_epochs}, Approximation Ratio:{avg_approx_ratio:.4f}")

        # # Early stopping based on approximation ratio improvement
        # if abs(avg_approx_ratio - 1) < abs(best_approx_ratio - 1):
        #     best_approx_ratio = avg_approx_ratio
        #     wait = 0
        #     torch.save(model, f"trained_model_1_{data_type}_{num_items}.pth") # Save best model
        # else:
        #     wait += 1
        #     if wait >= max_wait:
        #         print(f"Early stopping at epoch {epoch + 1}. No improvement over 5 epochs.")
        #         break


    # Save the trained model
    torch.save(model, f"trained_model_1_{data_type}_{num_items}.pth")
    print("Model saved as 'trained_model_1.pth")

    # Exit visual plot
    # Show a pop-up dialog when training is complete
    response = messagebox.showinfo("Training Complete","Training has completed successfully.")

    if response == "ok":
        visualizer.close_knapsack_plot()

# Example Usage
# if __name__ == "__main__":
#     data_type = 'SC'  
#     num_instances = 1000  
#     num_items = 10  
#     value_weight_range = 100  
#     H = 100  
#     train_knapsack_solver(data_type, num_instances, num_items, value_weight_range, H)