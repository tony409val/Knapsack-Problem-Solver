import tkinter as tk
from tkinter import ttk
from model_1_trainer import *
from model_2_trainer import *
from model_3_model import *
from utils import *
from evaluation import *
from test import *


model_path = None # Global variable to store the model path

def main(action, selected_model, data_type, num_items, status_var, root):
    global model_path 

     # Update status for the current action
    if action == "train":
        status_var.set(f"Training {selected_model} with {data_type} and {num_items} items...")
    elif action == "load":
        status_var.set(f"Loading model...")
    elif action == "test":
        status_var.set(f"Testing new sample with {num_items} items using {selected_model}...")
    elif action == "evaluate":
        status_var.set(f"Evaluating model performance with {num_items} items...")
    root.update_idletasks()

    #Train the model
    if action == "train":
        try:
            if selected_model == "Model 1 - GRU Based":
                train_knapsack_solver(data_type, num_items,
                                        num_epochs=100,
                                        batch_size=100,
                                        learning_rate=0.004,
                                        max_wait=5)
            elif selected_model == "Model 2 - RL Transformer":
                train_transformer_model(data_type, num_items)
            elif selected_model == "Model 3 - DAO-SNPS":
                model = DAOSNPS(data_type, num_items, visual=True)
                model.initialize()
                values, weights, capacity, optimal_selection, optimal_value, = model.prepare_data()
                model.run(values, weights, capacity, optimal_selection, max_generations=500, migration_interval=100)
                model.comparison_window(values, weights, capacity, optimal_selection, optimal_value)

            status_var.set("Training completed and model saved.")
                    
        except Exception as e:
           print(e)
           status_var.set(f"Error during training: {e}")

        root.update_idletasks()

    # Load a model
    elif action == "load":
        try:
            model_path = load_model()
            status_var.set(f"Model {model_path} loaded successfully.")

        except Exception as e:
            status_var.set(f"Error loading model: {e}")

        root.update_idletasks()

    # Test the model
    elif action == "test":
        try:
            test_model(model_path, data_type, num_items)
            status_var.set("Test completed.")

        except Exception as e:
           status_var.set(f"Error during testing: {e}")

        root.update_idletasks()

    # Evaluate the model
    elif action == "evaluate":
        if selected_model == "Model 3 - DAO-SNPS":
            #try:
                model = DAOSNPS(data_type, num_items, visual=False)
                model.evaluate()
            #except Exception as e:
            #    status_var.set(f"Error during evaluation: {e}")
        else:
            try:
                evaluate_model(model_path, data_type, num_items)
                status_var.set("Evaluation completed.")

            except Exception as e:
                status_var.set(f"Error during evaluation: {e}")

        
        root.update_idletasks()

    else:
        status_var.set("Invalid action or missing model file.")

        root.update_idletasks()

# GUI Function
def create_gui():
    root = tk.Tk()
    root.title("0/1 Knapsack Problem Solver")

    # Create and store input variables and set default values
    num_items_var = tk.StringVar(value="10") 
    data_type_var=tk.StringVar(value="UC") 
    algorithm_var = tk.StringVar(value="Model 1 - GRU Based")
    status_var = tk.StringVar(value="Ready") # Status message default   
    
    # Create GUI elements for user input

    # Item Volume Selection Label
    ttk.Label(root, text="Select Item Volume:").pack(pady=10, padx=200)
    num_items_dropdown = ttk.Combobox(root, textvariable=num_items_var, values=["5", "10", "20", "50", "100", "200"])
    num_items_dropdown.pack(pady=10)
    num_items_dropdown.set(num_items_var.get()) 

    # Instance Selection
    ttk.Label(root, text="Select Data Type:").pack(pady=10)
    data_type_dropdown = ttk.Combobox(root, textvariable=data_type_var, values=["UC", "SC", "SS"])
    data_type_dropdown.pack(pady=10)
    data_type_dropdown.set(data_type_var.get())

    # Algorithm Selection Label
    ttk.Label(root, text="Select Model:").pack(pady=10)
    algorithm_dropdown = ttk.Combobox(root, textvariable=algorithm_var, values=["Model 1 - GRU Based","Model 2 - RL Transformer", "Model 3 - DAO-SNPS"])
    algorithm_dropdown.pack(pady=10)
    algorithm_dropdown.set(algorithm_var.get())

# Training Start Button
    ttk.Button(
        root, 
        text="Train and Save Model", 
        command=lambda: main("train", algorithm_var.get(), data_type_var.get(), num_items_var.get(), status_var, root)
    ).pack(pady=10)

    # Load Trained Model Button
    ttk.Button(
        root, 
        text="Load Trained Model", 
        command=lambda: main("load", algorithm_var.get(), data_type_var.get(), num_items_var.get(), status_var, root)
    ).pack(pady=10)

    # Button for testing a new sample
    ttk.Button(
        root, 
        text="Test Random Sample", 
        command=lambda: main("test", algorithm_var.get(), data_type_var.get(), num_items_var.get(), status_var, root)
    ).pack(pady=10)

    # Button for evaluating a trained model
    ttk.Button(
        root,
        text="Evaluate Model",
        command=lambda: main("evaluate", algorithm_var.get(), data_type_var.get(), num_items_var.get(), status_var, root)
    ).pack(pady=10)

    # Label to display status messages
    ttk.Label(root, textvariable=status_var).pack(pady=50)
    
    # Tkinter loop start
    root.mainloop()
    print("Application Closed")

# GUI Call
create_gui()

