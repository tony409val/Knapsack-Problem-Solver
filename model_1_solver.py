import torch
import pickle
from model_1_utils import load_model

def greedy_decode(solution, weights, capacity):
    total_weight = sum(solution * weights)

    if total_weight <= capacity:
        return solution # Already feasible
    
    # Sort items by value-to-weight ratio and flip them one by one
    sorted_items = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)

    for idx in sorted_items:
        if solution[idx] == 1:
            solution[idx] == 0 # Flip the decision (remove the item)
            total_weight -= weights[idx]
            if total_weight <= capacity:
                break # Stop flipping when the solution becomes feasible
    
    return solution

# Inference function to solve new knapsack instances
def solve_knapsack(model, values, weights, capacity):
    model.eval() # Set the model to evaluation mode        

    # Get the model's prediction (binary probabilities)
    with torch.no_grad():
        probs = model(values, weights, capacity) # Output probs

    # Convert probabilities to a binary decision
    predicted_solution = [1 if p >= 0.5 else 0 for p in probs]

    # Check if the model's predicted solution exceeds capacity
    total_weight = sum(predicted_solution[i] * weights[i] for i in range(len(weights)))
    if total_weight > capacity:
        # If infeasible, use greedy decoding until the solution is feasible
        predicted_solution = greedy_decode(predicted_solution, weights, capacity)

    return predicted_solution

# Load instances from a saved file
def load_instances(instance_file):
    with open(instance_file, 'rb') as f:
        instances = pickle.load(f)
    return instances

# Main function to load model, run inference, and save solutions
# def main(model_path, instance_file, output_file):
#     # Load the model
#     model = load_model(model_path)

#     # Load instances for inference
#     instances = load_instances(instance_file)

#     # Solve the knapsack problem for the new instances
#     solutions = solve_knapsack(model, instances)

#     # Save the solutions
#     with open(output_file, 'wb') as f:
#         pickle.dump(solutions, f)

#     print(f"Solutions saved to {output_file}")


# Example Usage
# if __name__ == "__main__":
#     model_path = "trained_knapsack_model.pth"  # Path to the trained model
#     instance_file = "test_instances.pkl"  # Path to the test instances
#     output_file = "knapsack_solutions.pkl"  # Path to save the solutions

#     main(model_path, instance_file, output_file)