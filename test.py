import torch
import os
import random
import tkinter as tk
from tkinter import ttk
from model_1_utils import greedy_algorithm, load_data
from visual import *

# Function to flip decisions one by one until the solution is feasible
def greedy_decode(solution, weights, capacity):
    total_weight = sum(solution[i] * weights[i] for i in range(len(weights)))
    if total_weight <= capacity:
        return solution  # Already feasible

    # Sort items by weight and flip them one by one (remove largest weights first)
    sorted_items = sorted(range(len(weights)), key=lambda k: weights[k], reverse=True)

    for idx in sorted_items:
        if solution[idx] == 1:
            solution[idx] = 0  # Flip the decision
            total_weight -= weights[idx]
            if total_weight <= capacity:
                break  # Stop flipping when feasible

    return solution

# Function to test the model on a random instance of data
def test_model(model_path, data_type, num_items):

    # Load Evaluation data
    eval_file_name = f"model_1_eval_data_{data_type.lower()}_{num_items}.pkl"
    folder_path = "eval_data"
    file_path = os.path.join(folder_path, eval_file_name)

    if os.path.exists(file_path):
        eval_data = load_data(file_path)
    else:
        raise FileNotFoundError(f"Evaluation data file '{file_path}' not found.")

    # Load Solutions
    sol_file_name = f"model_1_eval_solutions_{data_type.lower()}_{num_items}.pkl"
    file_path = os.path.join(folder_path, sol_file_name)

    if os.path.exists(file_path):
        cbc_solutions = load_data(file_path)
    else:
        raise FileNotFoundError(f"Solutions file '{file_path}' not found.")
    
    # Randomly select an index
    random_idx = random.randint(0, len(eval_data) - 1)
    
    # Selected Instance and Solution through the random index
    selected_instance = eval_data[random_idx]
    cbc_solution = cbc_solutions[0][random_idx]

    # Load the model and set evaluation mode
    model = torch.load(model_path) # load the model
    model.eval() # Set model to evaluation mode

    # Initialize visual plot
    fig, ax = initialize_knapsack_plot()
    current_fig = fig

    with torch.no_grad():

        items, capacity = selected_instance
        values = [item[0] for item in items]
        weights = [item[1] for item in items]

        # CBC Optimal Solution
        cbc_value = cbc_solutions[1][random_idx]
        cbc_weight = sum(cbc_solution[i] * weights[i] for i in range(len(weights)))

        # Model's prediction
        model_input_values = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        model_input_weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)
        model_input_capacity = torch.tensor([capacity], dtype=torch.float32).unsqueeze(0).expand(-1, len(values))
        probs = model(model_input_values, model_input_weights, model_input_capacity).squeeze(0)

        # Convert probabilities to binary decisions
        predicted_solution = [1 if p >= 0.5 else 0 for p in probs]

        # Calculate predicted weight
        total_weight = sum(predicted_solution[i] * weights[i] for i in range(len(weights)))

        # Check if the model's prediction exceeds the capacity
        if total_weight > capacity:
            # Incrementally flip decisions if prediction is infeasible
            predicted_solution = greedy_decode(predicted_solution, weights, capacity)
            
            # Re-calculate total weight after greedy decoding
            total_weight = sum(predicted_solution[i] * weights[i] for i in range(len(weights)))

        # Greedy solution for comparison
        greedy_solution = greedy_algorithm(values, weights, capacity)
        greedy_weight = sum(greedy_solution[i] * weights[i] for i in range(len(weights)))

        # Calculate objective values
        predicted_value = sum(predicted_solution[i] * values[i] for i in range(len(values)))
        greedy_value = sum(greedy_solution[i] * values[i] for i in range(len(values)))

        # Approximation ratio relative to CBC optimal
        model_approx_ratio = predicted_value / cbc_value
        greedy_approx_ratio = greedy_value / cbc_value

        # Visualize results
        plot_knapsack(fig, ax, values, weights, predicted_solution, cbc_solution, capacity)

    # Display test results
    result_window = tk.Toplevel()
    result_window.title("Test Results")
    
    # Labels
    if int(num_items) <= 20:
        ttk.Label(result_window, text="Predicted Solution").grid(row=0, column=0, padx=10, pady=10)
        ttk.Label(result_window, text=f"Solution: {predicted_solution}, Total Weight: {total_weight}/{capacity}, Total Value: {predicted_value}, Approximation Ratio: {model_approx_ratio}").grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(result_window, text="Greedy Solution").grid(row=1, column=0, padx=10, pady=10)
        ttk.Label(result_window, text=f"Solution: {greedy_solution}, Total Weight: {greedy_weight}/{capacity}, Total Value: {greedy_value}, Approximation Ratio: {greedy_approx_ratio}").grid(row=1, column=1, padx=10, pady=10)

        ttk.Label(result_window, text="CBC Solution").grid(row=2, column=0, padx=10, pady=10)
        ttk.Label(result_window, text=f"Solution: {cbc_solution}, Total Weight: {cbc_weight}/{capacity}, Total Value: {cbc_value}").grid(row=2, column=1, padx=10, pady=10)
    else:
        ttk.Label(result_window, text="Predicted Solution").grid(row=0, column=0, padx=10, pady=10)
        ttk.Label(result_window, text=f"Total Weight: {total_weight}/{capacity}, Total Value: {predicted_value}, Approximation Ratio: {model_approx_ratio}").grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(result_window, text="Greedy Solution").grid(row=1, column=0, padx=10, pady=10)
        ttk.Label(result_window, text=f"Total Weight: {greedy_weight}/{capacity}, Total Value: {greedy_value}, Approximation Ratio: {greedy_approx_ratio}").grid(row=1, column=1, padx=10, pady=10)

        ttk.Label(result_window, text="CBC Solution").grid(row=2, column=0, padx=10, pady=10)
        ttk.Label(result_window, text=f"Total Weight: {cbc_weight}/{capacity}, Total Value: {cbc_value}").grid(row=2, column=1, padx=10, pady=10)


    # Close Button
    ttk.Button(result_window, text="Close", command=result_window.destroy).grid(row=3, column=1, padx=10, pady=10)

    result_window.mainloop()

