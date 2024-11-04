import matplotlib.pyplot as plt
import numpy as np
import torch

def initialize_knapsack_plot():
    """
    Initializes and returns a figure and axes for the knapsack plot.
    This function should be called once before training starts.
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    return fig, ax

def plot_knapsack(fig, ax, values, weights, model_solution, optimal_solution, capacity):
    """
    Plot a knapsack instance with items color-coded to show selections.
    
    Parameters:
    - values: Tensor or list of item values
    - weights: Tensor or list of item weights
    - model_solution: Tensor or list of model's predicted selection (binary list)
    - optimal_solution: Tensor or list of optimal selection (binary list)
    - capacity: Tensor or scalar of total capacity of the knapsack
    """
     
    # Convert tensors to lists if necessary
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    if isinstance(weights, torch.Tensor):
        weights = weights.tolist()
    if isinstance(model_solution, torch.Tensor):
        model_solution = model_solution.tolist() 
    if isinstance(optimal_solution, torch.Tensor):
        optimal_solution = optimal_solution.tolist()

    # Convert capacity to a scalar if it's a tensor
    if isinstance(capacity, torch.Tensor):
        capacity = capacity.item()

    # Clear previous data in the axes
    ax.clear()

    # Sort items by weight for a continuous line
    sorted_indices = np.argsort(weights)
    sorted_weights = np.array(weights)[sorted_indices]
    sorted_values = np.array(values)[sorted_indices]
    sorted_model_solution = np.array(model_solution)[sorted_indices]
    sorted_optimal_solution = np.array(optimal_solution)[sorted_indices]

    # Plot all items as gray dots for context
    ax.scatter(sorted_weights, sorted_values, color='gray', s=40, label='Items')

    # Model solution line
    model_selected_weights = sorted_weights[sorted_model_solution == 1]
    model_selected_values = sorted_values[sorted_model_solution == 1]
    ax.scatter(model_selected_weights, model_selected_values, color='gray', edgecolor='blue', s=200, label='Model Selection')
    # Optimal solution line
    optimal_selected_weights = sorted_weights[sorted_optimal_solution == 1]
    optimal_selected_values = sorted_values[sorted_optimal_solution == 1]
    ax.scatter(optimal_selected_weights, optimal_selected_values, color='gray', edgecolor='red', s=100, label='Optimal Selection')
    # Set plot limits
    ax.set_xlim(0, max(weights) + 5)
    ax.set_ylim(0, max(values) + 5)

    # Labels and legend
    ax.set_xlabel("Weight")
    ax.set_ylabel("Value")
    ax.set_title("Knapsack Item Selection Visualization")
    ax.legend(loc="upper left")

    # Calculate totals for model and optimal solutions
    model_value = sum(v for v, selected in zip(values, model_solution) if selected)
    model_weight = sum(w for w, selected in zip(weights, model_solution) if selected)
    optimal_value = sum(v for v, selected in zip(values, optimal_solution) if selected)
    optimal_weight = sum(w for w, selected in zip(weights, optimal_solution) if selected)

    # Display total values and weights
    fig.suptitle(
        f"Model: Value={model_value}, Weight={model_weight}/{capacity}   |   "
        f"Optimal: Value={optimal_value}, Weight={optimal_weight}/{capacity}",
        x=0.5, ha="center", fontsize=10
    )

    # Update the plot display
    fig.canvas.draw()
    fig.canvas.flush_events()

def close_knapsack_plot(fig):
    """
    Closes the knapsack plot at the end of training to free up resources.
    """
    plt.ioff()  # Turn off interactive mode
    plt.close(fig)