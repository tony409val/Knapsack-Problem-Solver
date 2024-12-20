import torch
import os
import time
import numpy as np
import  tkinter as tk
from tkinter import ttk
from utils import greedy_algorithm, load_data
from model_2_model import TransformerKnapsackModel
from model_1_model import NeuralKnapsackSolver
from visual import KnapsackVisualizer

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

# Function to evaluate the model on test data
def evaluate_model(model_path, data_type, num_items):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    infeasible_count = 0
    total_instances = len(eval_data)
    total_approx_ratio_model = 0
    total_approx_ratio_greedy = 0
    optimal_instances_model = 0
    optimal_instances_greedy = 0
    runtimes = []

    # Load the model and set it to evaluation mode
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for idx, (items, capacity) in enumerate(eval_data):
            values = [item[0] for item in items]
            weights = [item[1] for item in items]

            # CBC optimal solution
            cbc_solution = cbc_solutions[0][idx]
            cbc_value = sum(cbc_solution[i] * values[i] for i in range(len(values)))

            # Model's prediction
            model_input_values = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
            model_input_weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)
            model_input_capacity = torch.tensor([capacity], dtype=torch.float32).unsqueeze(0).expand(-1, len(values))
            
            # Check model type and adjust input accordingly
            if isinstance(model, TransformerKnapsackModel):
                # Create combined input for TransformerKnapsackModel
                model_input_combined = torch.cat(
                    (model_input_capacity.unsqueeze(2),
                     model_input_weights.unsqueeze(2),
                     model_input_values.unsqueeze(2),
                     torch.zeros(1, len(values), 3)  # placeholder for additional fields, adjust as necessary
                    ), dim=2
                )  # Shape: (1, num_items, 6)

                start_time = time()

                output = model(model_input_combined).squeeze(0)

                runtime = time() - start_time
                runtimes.append(runtime)

                # Interpret the output as Q-values, selecting the action with the highest Q-value
                predicted_solution = output.argmax(dim=1).tolist() # Use argmax along last dimension

                # Calculate predicted weight
                total_weight = sum(predicted_solution[i] * weights[i] for i in range(len(weights)))

                # Check if the model's prediction exceeds the capacity
                if total_weight > capacity:
                    infeasible_count += 1

            elif isinstance(model, NeuralKnapsackSolver):
                start_time = time()

                # Use input format for other models
                probs = model(model_input_values, model_input_weights, model_input_capacity).squeeze(0)

                runtime = time() - start_time
                runtimes.append(runtime)

                # Interpret the output as probabilities, using the threshold
                predicted_solution = [1 if p >= 0.5 else 0 for p in probs]
            
                # Calculate predicted weight
                total_weight = sum(predicted_solution[i] * weights[i] for i in range(len(weights)))

                # Check if the model's prediction exceeds the capacity
                if total_weight > capacity:
                    infeasible_count += 1
                    # Incrementally flip decisions if prediction is infeasible
                    predicted_solution = greedy_decode(predicted_solution, weights, capacity)

                    # Re-calculate total weight after greedy decoding
                    total_weight = sum(predicted_solution[i] * weights[i] for i in range(len(weights)))

            # Greedy solution for comparison
            greedy_solution = greedy_algorithm(values, weights, capacity)

            # Calculate objective values
            predicted_value = sum(predicted_solution[i] * values[i] for i in range(len(values)))
            greedy_value = sum(greedy_solution[i] * values[i] for i in range(len(values)))

            # Approximation ratio relative to CBC optimal
            model_approx_ratio = predicted_value / cbc_value
            greedy_approx_ratio = greedy_value / cbc_value

            total_approx_ratio_model += model_approx_ratio
            total_approx_ratio_greedy += greedy_approx_ratio

            # Check if the model's solution matches the CBC optimal solution
            if predicted_solution == cbc_solution:
                optimal_instances_model += 1

            # Check if the greedy solution matches the CBC optimal solution
            if greedy_solution == cbc_solution:
                optimal_instances_greedy += 1

    # Calculate average metrics
    avg_approx_ratio_model = total_approx_ratio_model / total_instances
    avg_approx_ratio_greedy = total_approx_ratio_greedy / total_instances
    infeasibility_rate = infeasible_count / total_instances
    optimal_instances_rate_model = optimal_instances_model / total_instances
    optimal_instances_rate_greedy = optimal_instances_greedy / total_instances
    avg_runtime = np.mean(runtimes)

    # Show evaluation results

    # Use visualizer to show the average approximation ratios
    visualizer = KnapsackVisualizer(approx_plot=True)
    visualizer.update_approx_plot([avg_approx_ratio_model, avg_approx_ratio_greedy])

    # Labels
    result_window = tk.Toplevel()
    result_window.title("Evaluation Results")

    ttk.Label(result_window, text=f"Number of instances: {len(eval_data)}"
             ).grid(row=0, column=0, padx=10, pady=10)
    ttk.Label(result_window, text=f"Infeasibility Rate: {infeasibility_rate}"
             ).grid(row=1, column=0, padx=10, pady=10)
    ttk.Label(result_window, text=f"Model Approximation Ratio: {avg_approx_ratio_model}"
             ).grid(row=2, column=0, padx=10, pady=10)
    ttk.Label(result_window, text=f"Greedy Approximation Ratio: {avg_approx_ratio_greedy}"
             ).grid(row=3, column=0, padx=10, pady=10)
    ttk.Label(result_window, text=f"Model Optimal Instances Rate: {optimal_instances_rate_model}"
             ).grid(row=4, column=0, padx=10, pady=10)
    ttk.Label(result_window, text=f"Greedy Optimal Instances Rate: {optimal_instances_rate_greedy}"
             ).grid(row=5, column=0, padx=10, pady=10)
    ttk.Label(result_window, text=f"Model Average Runtime: {avg_runtime}"
             ).grid(row=6, column=0, padx=10, pady=10)


    # Close Button
    ttk.Button(result_window, 
                text="Close", 
                command=lambda: (visualizer.close_knapsack_plot(), result_window.destroy())
                ).grid(row=6, column=0, padx=10, pady=10)
    
    result_window.mainloop()

# Example Usage

# if __name__ == "__main__":
#     model_path = "trained_knapsack_model.pth"
#     data_type = "UC"
#     num_items = "10"
    
#     evaluate_model(model_path, data_type, num_items)