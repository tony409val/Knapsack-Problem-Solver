import pulp
import random
import pickle

# Generate knapsack instances with increasing difficulty in batches of 100
def generate_knapsack_instances(instance_type, num_instances, num_items, R=1000, H=100):
    instances = []

    # Looper over the number of instances in steps of 100
    for step in range(0,num_instances, H):

        # Generate 100 instances with progressively increasing capacities
        for ins in range(H):
            items = []
            total_weight = 0

            # Generate items based on the instance type
            for _ in range(num_items):
                
                if instance_type == "UC":
                    # Uncorrelated: Values and weights uniformly distributed over [1, R]
                    value = random.randint(1, R)
                    weight = random.randint(1, R)
                elif instance_type == "SC":
                    # Stronlgy Correlated: value = weight + R/10
                    weight = random.randint(1, R)
                    value = weight + R // 10 
                elif instance_type == "SS":
                    # Subset Sum: value equals weight
                    weight = random.randint(1, R)
                    value = weight
                
                items.append((value, weight))
                total_weight += weight

            # Calculate capacity using the formula c = (h / (H + 1)) * total_weight
            # where h = current instance and H=100 (instances_per_step)
            h = ins + 1
            capacity = int((h / (H + 1)) * total_weight)

            # Add the instance (items and capacity) to the list
            instances.append((items, capacity))

    return instances


def solve_knapsack_instances(instances):
    solutions = []
    
    for items, capacity in instances:

        # Define the problem
        prob = pulp.LpProblem("KnapsackProblem", pulp.LpMaximize)

        # Define decision variables
        item_vars = [pulp.LpVariable(f'x{i}', cat='Binary') for i in range(len(items))]

        # Define the objective function (maximize total value)
        prob += pulp.lpSum([items[i][0] * item_vars[i] for i in range(len(items))])

        # Define the constraint (total weight <= capacity)
        prob += pulp.lpSum([items[i][1] * item_vars[i] for i in range(len(items))]) <= capacity

        # Solve the problem
        prob.solve()

        # Collect the solution
        solution = [int(item_vars[i].varValue) for i in range(len(items))]
        solutions.append(solution)

    return solutions


def generate_and_solve_instances(instance_type, num_instances, num_items, R, H=100):
    # Generate instances
    instances = generate_knapsack_instances(instance_type, num_instances, num_items, R, H)

    # Solve the instances to get the target solution
    solutions = solve_knapsack_instances(instances)

    # Format the data for training
    data = []
    for (items, capacity), solution in zip(instances, solutions):
        # Each instance is a tuple of (items, capacity, solution)
        # items: list of (value, weight) tuples
        # capacity: integer
        # solution: binary list indicating which items to inlcude
        data.append((items, capacity, solution))

    return data


def save_training_data(data, filename='training_data.pk1'):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f" Data saved to {filename}")


# Example Usage

if __name__ == "__main__":
    # Parameters for data generation
    # num_instances = 100  # Number of instances to generate
    # num_items = 10  # Number of items in each instance
    # R = 100  # Range for values and weights (R)
    # H = 100  # The constant H for capacity calculation

    # Generate and solve instances for Uncorrelated (UC) type
    # data_uc = generate_and_solve_instances('UC', num_instances, num_items, R, H)

    # TRAINING DATA
    data_uc_5 = generate_and_solve_instances('UC', 200000, 5, 1000, 100)
    data_uc_10 = generate_and_solve_instances('UC', 200000, 10, 1000, 100)
    data_uc_20 = generate_and_solve_instances('UC', 200000, 20, 1000, 100)
    data_uc_50 = generate_and_solve_instances('UC', 200000, 50, 1000, 100)
    data_uc_100 = generate_and_solve_instances('UC', 200000, 100, 1000, 100)
    data_uc_200 = generate_and_solve_instances('UC', 200000, 20, 1000, 100)


    data_sc_5 = generate_and_solve_instances('SC', 200000, 5, 1000, 100)
    data_sc_10 = generate_and_solve_instances('SC', 200000, 10, 1000, 100)
    data_sc_20 = generate_and_solve_instances('SC', 200000, 20, 1000, 100)
    data_sc_50 = generate_and_solve_instances('SC', 200000, 50, 1000, 100)
    data_sc_100 = generate_and_solve_instances('SC', 200000, 100, 1000, 100)
    data_sc_200 = generate_and_solve_instances('SC', 200000, 200, 1000, 100)


    data_ss_5 = generate_and_solve_instances('SS', 200000, 5, 1000, 100)
    data_ss_10 = generate_and_solve_instances('SS', 200000, 10, 1000, 100)
    data_ss_20 = generate_and_solve_instances('SS', 200000, 20, 1000, 100)
    data_ss_50 = generate_and_solve_instances('SS', 200000, 50, 1000, 100)
    data_ss_100 = generate_and_solve_instances('SS', 200000, 100, 1000, 100)
    data_ss_200 = generate_and_solve_instances('SS', 200000, 200, 1000, 100)

    # EVALUATION DATA
    eval_data_uc_5 = generate_knapsack_instances('UC', 1000, 5, 1000, 100)
    eval_data_uc_10 = generate_knapsack_instances('UC', 1000, 10, 1000, 100)
    eval_data_uc_20 = generate_knapsack_instances('UC', 1000, 20, 1000, 100)
    eval_data_uc_50 = generate_knapsack_instances('UC', 100, 50, 1000, 100)
    eval_data_uc_100 = generate_knapsack_instances('UC', 100, 100, 1000, 100)
    eval_data_uc_200 = generate_knapsack_instances('UC', 100, 200, 1000, 100)

    eval_data_sc_5 = generate_knapsack_instances('SC', 1000, 5, 1000, 100)
    eval_data_sc_10 = generate_knapsack_instances('SC', 1000, 10, 1000, 100)
    eval_data_sc_20 = generate_knapsack_instances('SC', 1000, 20, 1000, 100)
    eval_data_sc_50 = generate_knapsack_instances('SC', 100, 50, 1000, 100)
    eval_data_sc_100 = generate_knapsack_instances('SC', 100, 100, 1000, 100)
    eval_data_sc_200 = generate_knapsack_instances('SC', 100, 200, 1000, 100)

    eval_data_ss_5 = generate_knapsack_instances('SS', 1000, 5, 1000, 100)
    eval_data_ss_10 = generate_knapsack_instances('SS', 1000, 10, 1000, 100)
    eval_data_ss_20 = generate_knapsack_instances('SS', 1000, 20, 1000, 100)
    eval_data_ss_50 = generate_knapsack_instances('SS', 100, 50, 1000, 100)
    eval_data_ss_100 = generate_knapsack_instances('SS', 100, 100, 1000, 100)
    eval_data_ss_200 = generate_knapsack_instances('SS', 100, 200, 1000, 100)


    # Save data to a file
    #save_training_data(data_uc, "model_uc_training_data.pk1")
    save_training_data(data_uc_5, 'model_1_training_data_uc_5.pk1')
    save_training_data(data_uc_10, 'model_1_training_data_uc_10.pk1')
    save_training_data(data_uc_20, 'model_1_training_data_uc_20.pk1')
    save_training_data(data_uc_50, 'model_1_training_data_uc_50.pk1')
    save_training_data(data_uc_100, 'model_1_training_data_uc_100.pk1')
    save_training_data(data_uc_200, 'model_1_training_data_uc_200.pk1')

    save_training_data(data_sc_5, 'model_1_training_data_sc_5.pk1')
    save_training_data(data_sc_10, 'model_1_training_data_sc_10.pk1')
    save_training_data(data_sc_20, 'model_1_training_data_sc_20.pk1')
    save_training_data(data_sc_50, 'model_1_training_data_sc_50.pk1')
    save_training_data(data_sc_100, 'model_1_training_data_sc_100.pk1')
    save_training_data(data_sc_200, 'model_1_training_data_sc_200.pk1')

    save_training_data(data_ss_5, 'model_1_training_data_ss_5.pk1')
    save_training_data(data_ss_10, 'model_1_training_data_ss_10.pk1')
    save_training_data(data_ss_20, 'model_1_training_data_ss_20.pk1')
    save_training_data(data_ss_50, 'model_1_training_data_ss_50.pk1')
    save_training_data(data_ss_100, 'model_1_training_data_ss_100.pk1')
    save_training_data(data_ss_200, 'model_1_training_data_ss_200.pk1')

    save_training_data(eval_data_uc_5, 'model_1_eval_data_uc_5.pk1')
    save_training_data(eval_data_uc_10, 'model_1_eval_data_uc_10.pk1')
    save_training_data(eval_data_uc_20, 'model_1_eval_data_uc_20.pk1')
    save_training_data(eval_data_uc_50, 'model_1_eval_data_uc_50.pk1')
    save_training_data(eval_data_uc_100, 'model_1_eval_data_uc_100.pk1')
    save_training_data(eval_data_uc_200, 'model_1_eval_data_uc_200.pk1')

    save_training_data(eval_data_sc_5, 'model_1_eval_data_sc_5.pk1')
    save_training_data(eval_data_sc_10, 'model_1_eval_data_sc_10.pk1')
    save_training_data(eval_data_sc_20, 'model_1_eval_data_sc_20.pk1')
    save_training_data(eval_data_sc_50, 'model_1_eval_data_sc_50.pk1')
    save_training_data(eval_data_sc_100, 'model_1_eval_data_sc_100.pk1')
    save_training_data(eval_data_sc_200, 'model_1_eval_data_sc_200.pk1')

    save_training_data(eval_data_ss_5, 'model_1_eval_data_ss_5.pk1')
    save_training_data(eval_data_ss_10, 'model_1_eval_data_ss_10.pk1')
    save_training_data(eval_data_ss_20, 'model_1_eval_data_ss_20.pk1')
    save_training_data(eval_data_ss_50, 'model_1_eval_data_ss_50.pk1')
    save_training_data(eval_data_ss_100, 'model_1_eval_data_ss_100.pk1')
    save_training_data(eval_data_ss_200, 'model_1_eval_data_ss_200.pk1')




#     # Output some generated data for verification
#     print(f"Example Uncorrelated (UC) instance:\n{data_uc[0]}")

#     # You can repeat the process for Strongly Correlated (SC) and Subset Sum (SS)
#     # data_sc = generate_and_solve_instances('SC', num_instances, num_items, R, H)
#     # data_ss = generate_and_solve_instances('SS', num_instances, num_items, R, H)