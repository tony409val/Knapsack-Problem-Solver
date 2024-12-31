import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import calc_approx_ratio, calc_reward_approx

class KnapsackVisualizer:
    def __init__(self, knapsack_plot=False, approx_plot=False, reward_plot=False):
        self.is_paused = False
        self.knapsack_plot = knapsack_plot
        self.x_gap = 0
        
        plt.ion()  # Turn on interactive mode

        # Initialize primary plot
        if knapsack_plot:
            self.fig, self.ax = self._initialize_knapsack_plot()
            self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

            # Get the position of the knapsack plot window
            knapsack_x = self.fig.canvas.manager.window.winfo_x() 
            knapsack_width = self.fig.canvas.manager.window.winfo_width()

            # Calculate the position for the secondary plot (to the right of the knapsack plot)
            self.x_gap = knapsack_x + knapsack_width + 500  # Add a pixel gap
        
        # Only initialize specified secondary plots
        if approx_plot:
            self.approx_fig, self.approx_ax = self._initialize_approx_plot()
            self.approx_fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        else:
            self.approx_fig, self.approx_ax = None, None

        if reward_plot:
            self.reward_fig, self.reward_ax = self._initialize_reward_plot()
            self.reward_fig.canvas.mpl_connect("key_press_event", self._on_key_press)


        else:
            self.reward_fig, self.reward_ax = None, None
            
    def toggle_pause(self, event=None):
        """Toggle the pause state for the approximation plot."""
        self.is_paused = not self.is_paused
        print("Approximation plot paused" if self.is_paused else "Approximation plot resumed")

    def _on_key_press(self, event):
        """Handle key press events."""
        if event.key == ' ':  # Spacebar toggles pause
            self.toggle_pause()

    def _initialize_knapsack_plot(self):
        """
        Initializes and returns a figure and axes for the knapsack plot.
        This function should be called once before training starts.
        """
        fig, ax = plt.subplots()
        fig.canvas.manager.window.wm_geometry("+0+0") # Position knapsack plot on the left
        return fig, ax

    def _initialize_approx_plot(self):
        """
        Initializes and returns a figure and axes for the approximation ratio plot.
        This function should be called once before training starts.
        """

        fig, ax = plt.subplots() # Create a separate figure for the approx plot
        fig.canvas.manager.window.wm_geometry(f"+{self.x_gap}+0") # Position approx plot to the right of the knapsack plot
        ax.set_title("Average Approximation Ratio")
        ax.set_xlabel("Epoch/Episode/Generation")
        ax.set_ylabel("Approx Ratio")
        ax.legend("upper left")

        return fig, ax

    def _initialize_reward_plot(self):
        """
        Initializes and returns a figure and axes for the reward progress plot.
        This function should be called once before training starts.
        """
        
        fig, ax = plt.subplots()  # Create a separate figure for the reward plot
        fig.canvas.manager.window.wm_geometry(f"+{self.x_gap}+0") # Position reward plot to the right of the knapsack plot
        ax.set_title(f"Reward Average Approximation Ratio")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward Approx Ratio")
        ax.legend(loc="upper left")
        return fig, ax
        
    def update_approx_plot(self, avg_ratios):
        """
        Updates the average approximation progress plot.
        Shows a line plot if the input is a list of cumulative average ratios; shows a bar plot if the input is two final averages.

        Parameters:
        - avg_ratios: List of cumulative average ratios for line plot, or
                      List with exactly two values [avg_approx_ratio_model, avg_approx_ratio_greedy]
                      for final bar plot.
        """
        if len(avg_ratios) > 2:
            self.approx_fig.canvas.manager.window.wm_geometry(f"+{self.x_gap}+0") # Position reward plot to the right of the knapsack plot

            self.approx_ax.clear() # Clear the axes to redraw the updated ratios

            # Plot the cumulative average approximation ratios
            self.approx_ax.plot(avg_ratios, label="Average Approximation Ratio", color='blue')

            # Draw the optimal approximation ratio as a horizontal line
            self.approx_ax.axhline(y=1, label="Optimal Approx Ratio", color='green', linestyle="--")

            # Set custom y-axis limits to center the optimal approx ratio
            self.approx_ax.set_ylim(0, 2)

            self.approx_ax.set_title("Average Approximation Ratio")
            self.approx_ax.set_xlabel("Epoch/Episode/Generation")
            self.approx_ax.set_ylabel("Approx Ratio")
            self.approx_ax.legend(loc="upper left")

        elif len(avg_ratios) == 1:
            # Bar Plot for (average) approximation ratios

            # Bar plot to show the (average) ratios
            self.approx_ax.bar(
                ["Model"],  # Labels for the two bars
                avg_ratios,           # Heights for each bar
                color=['blue'],
                label=["Model (Average) Approx Ratio"]
            )

            # Draw the optimal approximation ratio as a horizontal line
            self.approx_ax.axhline(y=1, color='green', linestyle="--", label="Optimal Approx Ratio")

            # Set y-axis limits to center the optimal approximation ratio
            self.approx_ax.set_ylim(0, 2)

            # Titles and labels for bar plot
            self.approx_ax.set_title("Approximation Ratios")
            self.approx_ax.set_ylabel("Approx Ratio")
            self.approx_ax.legend(loc="upper left")

        else:
            raise ValueError("avg_ratios must be a list with either more than two values for a line plot, or exactly two values for a bar plot.")

        # Update the plot display
        self.approx_fig.canvas.draw()
        self.approx_fig.canvas.flush_events()

    def update_reward_plot(self, approx_ratios):
        """
        Updates the reward progress plot with the new rewards and optimal rewards.

        Parameters:
        - fig: The figure object for the reward approx plot
        - ax: The axes object for the reward approx plot
        - approx_ratios: List of  reward approximation ratios up to the current episode
        """

        self.reward_fig.canvas.manager.window.wm_geometry(f"+{self.x_gap}+0") # Position reward plot to the right of the knapsack plot


        self.reward_ax.clear()  # Clear the axes to redraw the updated rewards
        
        # Plot the approximation ratios
        self.reward_ax.plot(approx_ratios, label="Reward  Approximation Ratio", color='blue')

        # Draw the optimal reward as a horizontal line
        self.reward_ax.axhline(y=1, label="Optimal Reward", color='green', linestyle="--")


        # Set custom y-axis limits to center the optimal reward
        self.reward_ax.set_ylim(0, 2)  # Start from 0 and go up to twice the optimal reward

        self.reward_ax.set_title(f"Reward Average Approximation Ratio")
        self.reward_ax.set_xlabel("Episode")
        self.reward_ax.set_ylabel("Reward Approx Ratio")
        self.reward_ax.legend(loc="upper left")
        self.reward_fig.canvas.draw()
        self.reward_fig.canvas.flush_events()

    def plot_knapsack(self, values, weights, model_solution, capacity, optimal_solution):
        """
        Plot a knapsack instance with items color-coded to show selections.
        
        Parameters:
        - values: Tensor or list of item values
        - weights: Tensor or list of item weights
        - model_solution: Tensor or list of model's predicted selection (binary list)
        - optimal_solution: Tensor or list of optimal selection (binary list)
        - capacity: Tensor or scalar of total capacity of the knapsack
        """

        # Update plot if not paused
        if not self.is_paused:
        
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
            self.ax.clear()

            # Sort items by weight for a continuous line
            sorted_indices = np.argsort(weights)
            sorted_weights = np.array(weights)[sorted_indices]
            sorted_values = np.array(values)[sorted_indices]
            sorted_model_solution = np.array(model_solution)[sorted_indices]
            sorted_optimal_solution = np.array(optimal_solution)[sorted_indices] # Sort by weight 

            # not selected and not optimal items (gray dots)
            normal_indices = (sorted_model_solution == 0) & (sorted_optimal_solution == 0)
            self.ax.scatter(
                sorted_weights[normal_indices], 
                sorted_values[normal_indices], 
                color='gray', 
                s=100, 
                label='Not Selected Not Optimal'
            )

            # Optimal but not selected items (red bodied dots)
            optimal_not_selected_indices = (sorted_model_solution == 0) & (sorted_optimal_solution == 1)
            self.ax.scatter(
                sorted_weights[optimal_not_selected_indices], 
                sorted_values[optimal_not_selected_indices], 
                color='red', 
                s=100, 
                label='Optimal but Not Selected'
            )

            # Selected but not optimal items (grey body with blue edge color)
            selected_not_optimal_indices = (sorted_model_solution == 1) & (sorted_optimal_solution == 0)
            self.ax.scatter(
                sorted_weights[selected_not_optimal_indices], 
                sorted_values[selected_not_optimal_indices], 
                color='gray', 
                edgecolor='blue', 
                s=100, 
                label='Selected but Not Optimal'
            )

            # Optimal and selected items (red bodied with blue edge)
            optimal_and_selected_indices = (sorted_model_solution == 1) & (sorted_optimal_solution == 1)
            self.ax.scatter(
                sorted_weights[optimal_and_selected_indices], 
                sorted_values[optimal_and_selected_indices], 
                color='red', 
                edgecolor='blue', 
                s=100, 
                label='Optimal and Selected'
            )            
            # Set plot limits
            self.ax.set_xlim(0, max(weights) + 200)
            self.ax.set_ylim(0, max(values) + 200)

            # Labels and legend
            self.ax.set_xlabel("Weight")
            self.ax.set_ylabel("Value")
            self.ax.set_title("Knapsack Item Selection Visualization")
            self.ax.legend(loc="upper left")

            # Calculate totals for model and optimal solutions
            model_value = sum(v for v, selected in zip(values, model_solution) if selected)
            model_weight = sum(w for w, selected in zip(weights, model_solution) if selected)
            optimal_value = sum(v for v, selected in zip(values, optimal_solution) if selected)
            optimal_weight = sum(w for w, selected in zip(weights, optimal_solution) if selected)

            # Display total values and weights
            self.fig.suptitle(
                f"Model: Value={model_value}, Weight={model_weight}/{capacity}   |   "
                f"Optimal: Value={optimal_value}, Weight={optimal_weight}/{capacity}",
                x=0.5, ha="center", fontsize=10
            )

            # Update the plot display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close_knapsack_plot(self):

        """
        Closes the knapsack plot at the end of training to free up resources.
        """
        plt.ioff()  # Turn off interactive mode
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        if hasattr(self, 'approx_fix'):
            plt.close(self.approx_fig)
        if hasattr(self, 'reward_fig'):
            plt.close(self.reward_fig)

