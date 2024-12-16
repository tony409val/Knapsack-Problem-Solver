import numpy as np
import os
import random
import tkinter as tk
from tkinter import ttk
from utils import load_data, calc_approx_ratio, greedy_algorithm
from model_3_fitness_evaluation import *
from visual import KnapsackVisualizer


class SpikingNeuralPSystem:
    def __init__(self, num_neurons, problem_length):
        self.num_neurons = num_neurons
        self.problem_length = problem_length
        self.prob_matrix = np.random.uniform(0.05, 0.2, (num_neurons, problem_length))
        self.binary_matrix = np.zeros((num_neurons, problem_length), dtype=int)
        self.fitness = 0

    def initialize(self):
        self.binary_matrix = np.random.randint(0, 2, (self.num_neurons, self.problem_length))

    def update_probabilities(self, target_values, n_0, n_1, n):
        """
        Update the probabilities in the prob_matrix based on the target values and allele counts.

        Args:
        target_values (np.ndarray): Binary matrix of the best-performing individual.
        n_0 (np.ndarray): Number of zeros for each decision variable across the population.
        n_1 (np.ndarray): Number of ones for each decision variable across the population.
        n (int): Total number of individuals in the subpopulation.
        """
        # Adaptive learning weights
        w_0 = (n_0 + n) / (2 * n)
        w_1 = (n_1 + n) / (2 * n)
        
        # Update probabilities
        for i in range(self.num_neurons):
            for j in range(self.problem_length):
                if target_values[j] == 1:
                    self.prob_matrix[i, j] += w_0[j] * (1 - self.prob_matrix[i, j])
                else:
                    self.prob_matrix[i, j] += w_1[j] * (0 - self.prob_matrix[i, j])

    def calculate_fitness(self, values, weights, capacity):
        # Use the centralized batch fitness function
        self.fitness = evaluate_fitness_batch(values, weights, capacity, self.binary_matrix)

class SubPopulation:
    def __init__(self, num_individuals, num_neurons, problem_length):
        self.individuals = [SpikingNeuralPSystem(num_neurons, problem_length) for _ in range(num_individuals)]

    def initialize(self):
        for ind_idx, individual in enumerate(self.individuals):
            individual.initialize()


    def evaluate_fitness(self, values, weights, capacity):
        for individual in self.individuals:
            individual.calculate_fitness(values, weights, capacity)

    def mutate(self, values, weights, capacity, mutation_prob):
        """
        Apply mutation to each individual's binary matrix with probability mutation_prob.

        Args:
            mutation_prob (float): Probability of mutating each bit.
        """
        for individual in self.individuals:
            for i in range(individual.num_neurons):
                for j in range(individual.problem_length):
                    if np.random.rand() < mutation_prob:
                        # Flip the bit
                        individual.binary_matrix[i, j] = 1 - individual.binary_matrix[i, j]

        # Recalculate fitness after mutation
        individual.calculate_fitness(values, weights, capacity)

    def update_probabilities(self):
        # Find the best-performing solution as target values
        best_individual = max(self.individuals, key=lambda ind: ind.fitness.max())
        
        # Select the row in the binary matrix with the highest fitness as the target values
        target_values_idx = np.argmax(best_individual.fitness)  # Index of the best row
        target_values = best_individual.binary_matrix[target_values_idx]  # Best row as 1D target array

        # Compute allele counts
        n = len(self.individuals)
        n_0 = np.sum([np.sum(ind.binary_matrix == 0, axis=0) for ind in self.individuals], axis=0)
        n_1 = np.sum([np.sum(ind.binary_matrix == 1, axis=0) for ind in self.individuals], axis=0)

        # Update probabilities for all individuals
        for individual in self.individuals:
            individual.update_probabilities(target_values, n_0, n_1, n)

class DAOSNPS:
    def __init__(self, data_type, num_items, visual=True):

        # Model Parameters
        self.num_subpopulations = 5
        self.num_individuals = 10
        self.num_neurons = int(num_items) + 2
        self.problem_length = int(num_items)
        self.mutation_probability = 0.01

        self.subpopulations = [SubPopulation(self.num_individuals, self.num_neurons, self.problem_length)
                               for _ in range(self.num_subpopulations)]
        self.global_best = None

        self.data_type = data_type
        self.num_items = num_items
        self.visual = visual

        # Initialize the visualizer
        if self.visual:
            self.visualizer = KnapsackVisualizer(knapsack_plot=True, approx_plot=True)
        else:
            self.visualizer = None

        
    def initialize(self):
        for subpopulation in self.subpopulations:
            subpopulation.initialize()

    def run(self, values, weights, capacity, optimal_selection,max_generations=500, migration_interval=100):

        approx_ratios = [] # To track approximatin ratios for visualization

        for generation in range(max_generations):
            print(f"Generation {generation + 1}:")
      
            # Evaluate fitness and update probabilities
            for sub_idx, subpopulation in enumerate(self.subpopulations):
                subpopulation.evaluate_fitness(values, weights, capacity)
                subpopulation.update_probabilities()  # Adjust probabilities after fitness evaluation
                subpopulation.mutate(values, weights, capacity, self.mutation_probability)

            # Log the best fitness in the subpopulation
            best_fitness = max([ind.fitness.max() for ind in subpopulation.individuals])
            print(f"  Subpopulation {sub_idx + 1} Best Fitness: {best_fitness}")

            # Update knapsack plot in real time
            if self.visualizer:
                best_individual = max(subpopulation.individuals, key=lambda ind: ind.fitness.max())
                max_row_idx = np.argmax(best_individual.fitness)
                model_solution = best_individual.binary_matrix[max_row_idx]

                self.visualizer.plot_knapsack(values, weights, model_solution, capacity, optimal_selection)

            # Update approximation ratio plot
            if self.visual:
                approx = calc_approx_ratio(model_solution, optimal_selection, values)
                approx_ratios.append(approx)
                if len(approx_ratios) > 2:
                    self.visualizer.update_approx_plot(approx_ratios)

            if generation % migration_interval == 0:
                self.perform_information_exchange()

    def perform_information_exchange(self):
        for i in range(len(self.subpopulations)):
            subpop_a = self.subpopulations[i]
            subpop_b = self.subpopulations[(i + 1) % len(self.subpopulations)]

            # Mean fitness for subpopulation A
            mean_fitness_a = np.mean([np.sum(ind.fitness) for ind in subpop_a.individuals])

            # Select migration candidate from subpopulation A
            candidate_set = [ind for ind in subpop_a.individuals if np.sum(ind.fitness) > mean_fitness_a]
            if not candidate_set:  # If no candidates, skip this exchange
                continue
            migration_candidate = max(candidate_set, key=lambda ind: np.sum(ind.fitness))

            # Select replacement candidate from subpopulation B
            replacement_candidates = sorted(subpop_b.individuals, key=lambda ind: np.sum(ind.fitness))
            replacement_candidate = replacement_candidates[0]

            # Perform migration
            subpop_b.individuals.remove(replacement_candidate)
            subpop_b.individuals.append(migration_candidate)

    def prepare_data(self):
        """
        Load pickle data file, extract and return values.

        Args:
            data_type (str):  data type to load ('UC','SS','SC').

        Returns:
            values, weights, capacity, optimal_selection, optimal_value
        """
        file_name = f"training_data_{self.data_type.lower()}_{self.num_items}.pkl"
        folder_path = "presentation_data"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            training_data = load_data(file_path)
        else:
            raise FileNotFoundError(f"Training data file '{file_path}' not found.")
        
        print(training_data)

        random_instance = random.choice(training_data)
        items = random_instance[0]
        capacity = random_instance[1]
        optimal_selection = random_instance[2]
        optimal_value = random_instance[3]

        values, weights = zip(*items)

        return np.array(values), np.array(weights), capacity, optimal_selection, optimal_value
    
    def comparison_window(self, values, weights, capacity, optimal_selection, optimal_value):

        """
        Compute and show the total value and weight of the model's best solution across all subpopulations,
        along with the greedy algorithm solution for comparison.

        Args:
            values (np.ndarray): Array of item values.
            weights (np.ndarray): Array of item weights.
            capacity (scalar): capacity of the knapsack
            optimal_selection (list): model solver optimal solution
        """
        model_value = 0
        model_weight = 0
        model_solution = None

        for subpopulation in self.subpopulations:
            for individual in subpopulation.individuals:
                # Find the best solution (row) in this individual's binary_matrix
                max_row_idx = np.argmax(individual.fitness)
                solution = individual.binary_matrix[max_row_idx]
                total_value = np.sum(solution * values)
                total_weight = np.sum(solution * weights)

                # Update the global best if this solution is better
                if total_value > model_value:
                    model_value = total_value
                    model_weight = total_weight
                    model_solution = solution

        # Print the best solution details
        print(f"Best Solution: {model_solution}")
        print(f"Total Value: {model_value}")
        print(f"Total Weight: {model_weight}")
        print(f"Capacity: {capacity}")

        # Greedy solution for comparison
        greedy_solution = greedy_algorithm(values, weights, capacity)
        greedy_weight = np.sum(greedy_solution * weights)
        greedy_value = np.sum(greedy_solution * values)

        # Optimal weight
        optimal_weight = np.sum(optimal_selection * weights)


        # Approximation ratio relative to the model Solver optimal
        model_approx_ratio = model_value / optimal_value
        greedy_approx_ratio = greedy_value / optimal_value

        print(f"approx ratio model: {model_approx_ratio}")

        # Display test results
        result_window = tk.Toplevel()
        result_window.title("Test Results")


            # Labels
        if int(len(values)) <= 20:
            ttk.Label(result_window, text="Predicted Solution").grid(row=0, column=0, padx=10, pady=10)
            ttk.Label(result_window, text=f"Solution: {model_solution}, Total Weight: {total_weight}/{capacity}, Total Value: {model_value}, Approximation Ratio: {model_approx_ratio}"
                      ).grid(row=0, column=1, padx=10, pady=10)

            ttk.Label(result_window, text="Greedy Solution").grid(row=1, column=0, padx=10, pady=10)
            ttk.Label(result_window, text=f"Solution: {greedy_solution}, Total Weight: {greedy_weight}/{capacity}, Total Value: {greedy_value}, Approximation Ratio: {greedy_approx_ratio}"
                      ).grid(row=1, column=1, padx=10, pady=10)

            ttk.Label(result_window, text="Optimal Solution").grid(row=2, column=0, padx=10, pady=10)
            ttk.Label(result_window, text=f"Solution: {optimal_selection}, Total Weight: {optimal_weight}/{capacity}, Total Value: {optimal_value}"
                      ).grid(row=2, column=1, padx=10, pady=10)
        else:
            ttk.Label(result_window, text="Predicted Solution").grid(row=0, column=0, padx=10, pady=10)
            ttk.Label(result_window, text=f"Total Weight: {total_weight}/{capacity}, Total Value: {model_value}, Approximation Ratio: {model_approx_ratio}"
                      ).grid(row=0, column=1, padx=10, pady=10)

            ttk.Label(result_window, text="Greedy Solution").grid(row=1, column=0, padx=10, pady=10)
            ttk.Label(result_window, text=f"Total Weight: {greedy_weight}/{capacity}, Total Value: {greedy_value}, Approximation Ratio: {greedy_approx_ratio}"
                      ).grid(row=1, column=1, padx=10, pady=10)

            ttk.Label(result_window, text="Optimal Solution").grid(row=2, column=0, padx=10, pady=10)
            ttk.Label(result_window, text=f"Total Weight: {optimal_weight}/{capacity}, Total Value: {optimal_value}"
                      ).grid(row=2, column=1, padx=10, pady=10)


        # Close Button
        ttk.Button(result_window, 
                text="Close", 
                command=lambda: (self.visualizer.close_knapsack_plot(), result_window.destroy())
                ).grid(row=3, column=1, padx=10, pady=10)

        result_window.mainloop()

    def evaluate(self):

        import time
        runtimes = []
        infeasible_count = 0
        total_instances= 10
        total_approx_ratio_model = 0
        total_approx_ratio_greedy = 0

        # Initialize model
        self.initialize()

        for instance in range(total_instances):

            # Get a random instance
            values, weights, capacity, optimal_selection, optimal_value = self.prepare_data()

            # Solve the problem and measure runtime
            start_time = time.time()
            self.run(values=values, weights=weights, capacity=capacity, optimal_selection=optimal_selection)
            runtime = time.time() - start_time
            runtimes.append(runtime)

            # Get the model's best solution
            model_value = 0
            model_weight = 0
            model_solution = None

            for subpopulation in self.subpopulations:
                for individual in subpopulation.individuals:
                    # Find the best solution (row) in this individual's binary_matrix
                    max_row_idx = np.argmax(individual.fitness)
                    solution = individual.binary_matrix[max_row_idx]
                    total_value = np.sum(solution * values)
                    total_weight = np.sum(solution * weights)

                    # Update the global best if this solution is better
                    if total_value > model_value:
                        model_value = total_value
                        model_weight = total_weight
                        model_solution = solution
            
            # Update infeasibility rate
            if total_weight > capacity:
                infeasible_count += 1

            # Get greedy solution
            greedy_solution = greedy_algorithm(values, weights, capacity)
            greedy_value = np.sum(greedy_solution * values)


            # Compute model approx ratio
            approx_ratio = model_value / optimal_value
            total_approx_ratio_model += approx_ratio

            # Compute greedy approx value
            greedy_approx = greedy_value / optimal_value
            total_approx_ratio_greedy += greedy_approx

        # Compute metrics
        avg_model_approx = total_approx_ratio_model / total_instances
        avg_greedy_approx = total_approx_ratio_greedy / total_instances
        avg_runtime = np.mean(runtimes)
        infeasibility_rate = infeasible_count / total_instances

        # Use visualizer to show the average approximation ratios
        visualizer = KnapsackVisualizer(approx_plot=True)
        visualizer.update_approx_plot([avg_model_approx, avg_greedy_approx])

        # Labels
        result_window = tk.Toplevel()
        result_window.title("Evaluation Results")

        ttk.Label(result_window, text=f"Number of instances: {total_instances}").grid(row=0, column=0, padx=10, pady=10)
        ttk.Label(result_window, text=f"Infeasibility Rate: {infeasibility_rate}").grid(row=1, column=0, padx=10, pady=10)
        ttk.Label(result_window, text=f"Model Approximation Ratio: {avg_model_approx}").grid(row=2, column=0, padx=10, pady=10)
        ttk.Label(result_window, text=f"Greedy Approximation Ratio: {avg_greedy_approx}").grid(row=3, column=0, padx=10, pady=10)
        ttk.Label(result_window, text=f"Average runtime: {avg_runtime}sec").grid(row=4, column=0, padx=10, pady=10)
        
        # Close Button
        ttk.Button(result_window, 
                    text="Close", 
                    command=lambda: (visualizer.close_knapsack_plot(), result_window.destroy())
                    ).grid(row=6, column=0, padx=10, pady=10)
        
        result_window.mainloop()

# Test code

# data_type = 'UC'
# num_items = 100

# # DAOSNPS Parameters
# num_subpopulations = 5
# num_individuals = 10
# num_neurons = num_items + 2
# max_generations = 500
# migration_interval = 100
# mutation_prob = 0.01
# problem_length = num_items

# # Initialize model
# model = DAOSNPS(num_subpopulations, num_individuals, num_neurons, problem_length, mutation_prob)
# model.initialize()

# values, weights, capacity, optimal_selection, optimal_value = model.prepare_data(data_type, num_items)

# print(f"Values: {values}, Weights: {weights}, Capacity: {capacity}")


# model.run(values, weights, capacity, max_generations, migration_interval)
# model.print_model_solution(values, weights, capacity)