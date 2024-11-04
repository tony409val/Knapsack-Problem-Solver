import torch
import os
from model_1_utils import greedy_algorithm, load_data

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

    # Load the model and set it to evaluation mode
    model = torch.load(model_path)
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
            probs = model(model_input_values, model_input_weights, model_input_capacity).squeeze(0)

            # Convert probabilities to binary decisions
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

    return {
        'infeasibility_rate': infeasibility_rate,
        'avg_approx_ratio_model': avg_approx_ratio_model,
        'avg_approx_ratio_greedy': avg_approx_ratio_greedy,
        'optimal_instances_rate_model': optimal_instances_rate_model,
        'optimal_instances_rate_greedy': optimal_instances_rate_greedy
    }

# Example Usage

# if __name__ == "__main__":
#     model_path = "trained_knapsack_model.pth"
#     data_type = "UC"
#     num_items = "10"
    
#     evaluate_model(model_path, data_type, num_items)