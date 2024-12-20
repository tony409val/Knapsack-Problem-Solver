import torch
import pickle
import os
from tkinter import filedialog


### 1. Model Saving and Loading ###
def save_model(model, file_path):
    """Save the PyTorch model to a file."""
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model():
    """Load the PyTorch model from a file."""
    model_path = filedialog.askopenfilename(
            title="Select a Trained Model File", filetypes=[("Model Files", "*.pth")]
    )
    
    print(f"Model loaded from {model_path}")

    # Return model file path
    return model_path

### 2. Data Handling ###
def save_data(data, filename):
    """Save data to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

def load_data(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from {file_path}")
    
    return data

### 3. Greedy Algorithm ###
def greedy_algorithm(values, weights, capacity):

    """Greedy algorithm solution."""
    value_weight_ratio = [(values[i] / weights[i], i) for i in range(len(values))]
    value_weight_ratio.sort(reverse=True, key=lambda x: x[0])

    total_weight = 0
    solution = [0] * len(values)

    for ratio, index in value_weight_ratio:
        if total_weight + weights[index] <= capacity:
            solution[index] = 1
            total_weight += weights[index]

    return solution

### 4. Approximation Ratio Calculator ###
def calc_approx_ratio(predicted_solution, cbc_solution, values):
    # Ensure inputs are flattened lists
    if isinstance(predicted_solution, torch.Tensor):
        predicted_solution = predicted_solution.flatten().tolist()
    if isinstance(cbc_solution, torch.Tensor):
        cbc_solution = cbc_solution.flatten().tolist()
    if isinstance(values, torch.Tensor):
        values = values.flatten().tolist()

    """Calculate the approximation ratio for a predicted solution."""

    # Calculate the total value for the predicted and CBC solutions
    predicted_value = sum(predicted_solution[i] * values[i] for i in range(len(values)))
    cbc_value = sum(cbc_solution[i] * values[i] for i in range(len(values)))

    # Return the approximation ratio
    return predicted_value / cbc_value

### 4. Reward Approximation Ratio Calculator (unsupervised models) ###
def calc_reward_approx(rewards, optimal_rewards):
    """
    Calculates the cumulative average approximation ratio of model rewards to optimal rewards.

    Parameters:
    - rewards: List of model rewards for each episode
    - optimal_reward: List of optimal reward for each episode

    Returns:
    - List of cumulative average approximation ratios for each episode
    """
    # Calculate the approximation ratio for each episode
    approximation_ratios = [
        reward / optimal if optimal > 0 else 0
        for reward, optimal in zip(rewards, optimal_rewards)
    ]
    
    return approximation_ratios

### 5. Load and print Pickle file
def print_pickle(file_path, num_lines=2):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            
            # Print the first `num_lines` of data if it is iterable
            if isinstance(data, (list, tuple)):
                for i, line in enumerate(data):
                    if i >= num_lines:
                        break
                    print(line)
            else:
                print("Data is not iterable or does not contain multiple lines.")
    except Exception as e:
        print(f"An error occurred: {e}")

## Print pickle
# file_name = f"training_data_uc_100.pkl"  
# folder_path = 'presentation_data'
# file_path = os.path.join(folder_path, file_name)
# print(f"EVAL DATA----------------------")
# print_pickle(file_path)
# print(f"SOLUTIONS--------------------")

# file_name = f"model_1_eval_solutions_uc_10.pkl"  
# file_path = os.path.join(folder_path, file_name)
# data = load_data(file_path)
# print(f"Length of cbc_solutions: {len(data)}")
# print_pickle(file_path)
