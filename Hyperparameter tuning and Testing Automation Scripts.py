# HYPERPARAMETER TUNING SCRIPTS -------------------------------------
## HYPERPARAMETER TUNING DNN
def hyperparameter_tuning_nn(training_data, validation_data, optimal_value):
    param_grid = {
        'lr': [0.01, 0.001, 0.0001, 0.00001],
        'hidden_sizes': [[64, 64], [128, 128], [256, 128]],
        'epochs': [500, 1000]
    }

    best_params = None
    best_performance = float('inf')  # We want to minimize the absolute optimality gap

    for params in ParameterGrid(param_grid):
        model = KnapsackDNN(
            input_size=len(training_data[0][0]) + len(training_data[0][1]) + 1,
            hidden_sizes=params['hidden_sizes'],
            output_size=len(training_data[0][0])
        )

        train_nn(model, training_data, epochs=params['epochs'], lr=params['lr'])
        performance = evaluate_model_nn(model, validation_data, optimal_value)

        if performance < best_performance:  # We want the smallest absolute optimality gap
            best_performance = performance
            best_params = params

    print(f"Best Parameters: {best_params}")
    print(f"Best Performance (Optimality Gap): {best_performance}")

def evaluate_model_nn(model, test_data, optimal_value):
    total_value = 0
    total_weight = 0
    optimal_value = optimal_value  # This should be precomputed for your test data
    selected_items = []

    for sample in test_data:
        values = sample[0]
        weights = sample[1]
        capacity = sample[2]
        items = [(v, w) for v, w in zip(values, weights)]
        
        input_data = torch.FloatTensor(values + weights + [capacity])
        output = model(input_data).detach().numpy()

        selected = np.round(output).astype(int)
        selected_items.append(selected)
        
        value = sum(v for i, v in enumerate(values) if selected[i] == 1)
        weight = sum(w for i, w in enumerate(weights) if selected[i] == 1)

        total_value += value
        total_weight += weight

        optimality_gap = abs((optimal_value - total_value) / optimal_value)

    print(f"Total Value: {total_value}")
    print(f"Total Weight: {total_weight}")
    print(f"Optimal Value: {optimal_value}")
    print(f"Optimality Gap: {optimality_gap}")
    print(f"Selected Items: {selected_items}")

    return optimality_gap

### Place lines in def solve_knapsack accordingly
test_values = [random.randint(1, 50) for _ in range(num_items)]
test_weights = [random.randint(1, 20) for _ in range(num_items)]
test_knapsack_capacity = 50
solution = knapsack_dp(test_values, test_weights, test_knapsack_capacity)
optimal_value = solution[0][1]
validation_data.append((test_values, test_weights, test_knapsack_capacity))
hyperparameter_tuning_nn(training_data, validation_data, optimal_value)

## HYPERPARAMETER TUNING RNN
def evaluate_model_rnn(model, prepared_data, optimal_value):
    total_value = 0
    total_weight = 0
    selected_items = []

    model.eval()

    for input_tensors, label_tensors, capacity in prepared_data:
        input_tensors = input_tensors.to(next(model.parameters()).device)

        capacity_tensor = torch.tensor([capacity], dtype=torch.float32).unsqueeze(0).to(input_tensors.device)
        
        with torch.no_grad():
            output = model(input_tensors, capacity_tensor).cpu().numpy().flatten()

        selected = np.round(output).astype(int)
        selected_items.append(selected)
        
        values = input_tensors.squeeze(0)[:, 0].cpu().numpy()
        weights = input_tensors.squeeze(0)[:, 1].cpu().numpy()
        
        value = sum(v for i, v in enumerate(values) if selected[i] == 1)
        weight = sum(w for i, w in enumerate(weights) if selected[i] == 1)

        total_value += value
        total_weight += weight

    optimality_gap = (optimal_value - total_value) / optimal_value

    print(f"Total Value: {total_value}")
    print(f"Total Weight: {total_weight}")
    print(f"Optimal Value: {optimal_value}")
    print(f"Optimality Gap: {optimality_gap}")
    print(f"Selected Items: {selected_items}")

    return abs(optimality_gap)

def hyperparameter_tuning_rnn(prepared_data, validation_data, optimal_value):
    param_grid = {
        'lr': [0.01, 0.001, 0.0001, 0.00001],
        'hidden_sizes': [[64, 64], [128, 128], [256, 128]],
        'dropout_prob': [0.3, 0.5, 0.7],
        'num_layers': [1, 2, 3],
        'epochs': [500, 1000]
    }

    best_params = None
    best_performance = float('inf')  # We want to minimize the absolute optimality gap

    # Compute optimal value for validation data
    optimal_value = optimal_value

    for params in ParameterGrid(param_grid):
        model = KnapsackRNN(
            input_size=21,
            hidden_sizes=params['hidden_sizes'],
            output_size=10,
            num_layers=params['num_layers'],
            dropout_prob=params['dropout_prob']
        )

        train_rnn(model, prepared_data, epochs=params['epochs'], lr=params['lr'])
        performance = evaluate_model_rnn(model, validation_data, optimal_value)

        if performance < best_performance:  # We want the smallest absolute optimality gap
            best_performance = performance
            best_params = params

    print(f"Best Parameters: {best_params}")
    print(f"Best Performance (Optimality Gap): {best_performance}")

### Place lines in def solve_knapsack accordingly
test_values = [random.randint(1, 50) for _ in range(num_items)]
test_weights = [random.randint(1, 20) for _ in range(num_items)]
test_knapsack_capacity = 50
solution = knapsack_dp(test_values, test_weights, test_knapsack_capacity)
optimal_value = solution[0][1]
validation_data.append((test_values, test_weights, test_knapsack_capacity))
prepared_validation_data = prepare_rnn_data(validation_data)
hyperparameter_tuning_rnn(prepared_data, prepared_validation_data, optimal_value)

## DQN Hyperparameter tuning
def hyperparameter_tuning_dqn(training_data, validation_data, optimal_value):
    param_grid = {
        'lr': [0.01, 0.001, 0.0001],
        'hidden_sizes': [[64, 64], [128, 128], [256, 128]],
        'episodes': [500, 1000],  
        'batch_size': [32, 64, 128],
        'gamma': [0.9, 0.95, 0.99],
        'target_update': [5, 10, 20],
        'epsilon_decay': [0.99, 0.995, 0.999]
    }

    best_params = None
    best_performance = float('inf')  # We want to minimize the absolute optimality gap

    for params in ParameterGrid(param_grid):
        # Initialize the DQN agent with the current set of hyperparameters
        dqn_agent = DQNAgent(
            input_dim=4,  
            hidden_sizes=params['hidden_sizes'],
            output_dim=2,  
            lr=params['lr'],
            batch_size=params['batch_size'],
            gamma=params['gamma'],
            target_update=params['target_update'],
            decay_rate=params['epsilon_decay']
        )

        # Train the DQN agent
        train_dqn(dqn_agent, training_data, episodes=params['episodes'])

        # Evaluate the DQN agent on validation data using optimality gap
        performance = evaluate_model_dqn(dqn_agent, validation_data, optimal_value)

        if performance < best_performance:  # We want the smallest absolute optimality gap
            best_performance = performance
            best_params = params

    print(f"Best Parameters: {best_params}")
    print(f"Best Performance (Optimality Gap): {best_performance}")

def train_dqn(agent, training_data, epochs):
    agent.train_dqn(training_data, visualization=None, num_episodes=epochs)

def evaluate_model_dqn(agent, validation_data, optimal_value):
    total_value = 0
    total_weight = 0
    selected_items = []

    for sample in validation_data:
        values, weights, knapsack_capacity = sample[0], sample[1], sample[2]
        prediction = agent.solve_knapsack_with_dqn(values, weights, knapsack_capacity)
        
        selected = prediction[0][0]  # Get the selected items (actions taken by the agent)
        selected_items.append(selected)
        
        value = sum(v for i, v in enumerate(values) if selected[i] == 1)
        weight = sum(w for i, w in enumerate(weights) if selected[i] == 1)

        total_value += value
        total_weight += weight

        optimality_gap = abs((optimal_value - total_value) / optimal_value)

    print(f"Total Value: {total_value}")
    print(f"Total Weight: {total_weight}")
    print(f"Optimal Value: {optimal_value}")
    print(f"Optimality Gap: {optimality_gap}")
    print(f"Selected Items: {selected_items}")

    return optimality_gap

### Place Lines in def solve_knapsack accordingly
test_values = [random.randint(1, 50) for _ in range(num_items)]
test_weights = [random.randint(1, 20) for _ in range(num_items)]
test_knapsack_capacity = 50
solution = knapsack_dp(test_values, test_weights, test_knapsack_capacity)
optimal_value = solution[0][1]
validation_data.append((test_values, test_weights, test_knapsack_capacity))
hyperparameter_tuning_dqn(training_data, validation_data, optimal_value)

# TESTING AUTOMATION SCRIPTS -------------------------------------------------------------

## Function to train and evaluate a model
def train_and_evaluate_model(model, data, train_func, solve_func, num_items):
    start_time = time()

    # Train the model
    loss_values, memory_usage =  train_func(model, data, None, epochs=1000)

    end_time = time.time()
    total_time = end_time - start_time

    # Evaluate the model
    correct_predictions = 0
    total_items = 0

    for values, weights, knapsack_capacity, selected_items in data:
        prediction = solve_func(model, values, weights, knapsack_capacity, num_items)
        correct_predictions += sum([1 if p == s else 0 for p, s in zip(prediction[0][0], selected_items)])
        total_items += len(selected_items)

    accuracy = correct_predictions / total_items

    return accuracy, total_time, loss_values, memory_usage

## Function to automate the entire process
def automate_process(item_volumes, num_samples_per_volume):
    results = defaultdict(lambda: defaultdict(list))

    for num_items in item_volumes:
            print(f"Processing item volume: {num_items}")

            # Generate dataset for this item volume
            data = generate_data(num_samples_per_volume, num_items)

            # Initialize models (with best hyperparameters from Optuna)
            dnn_model = KnapsackDNN(2 * num_items + 1, [256, 128], num_items)
            rnn_model = KnapsackRNN(2 * num_items + 1, [64, 64], num_items, num_layers=1)
            dqn_model = DQNAgent(input_dim=4, hidden_sizes=[256, 128], output_dim=2)

            # Train and evaluate DNN
            dnn_accuracy, dnn_time, dnn_loss, dnn_memory = train_and_evaluate_model(dnn_model, data, train_nn, solve_knapsack_with_nn, num_items)
            results[num_items]['DNN_Accuracy'].append(dnn_accuracy)
            results[num_items]['DNN_Training_Time'].append(dnn_time)
            results[num_items]['DNN_Loss'].append(dnn_loss)
            results[num_items]['DNN_Memory_Usage'].append(dnn_memory)

            # Train and evaluate RNN
            prepared_data = prepare_rnn_data(data)
            rnn_accuracy, rnn_time, rnn_loss, rnn_memory = train_and_evaluate_model(rnn_model, prepared_data, train_rnn, solve_knapsack_with_rnn, num_items)
            results[num_items]['RNN_Accuracy'].append(rnn_accuracy)
            results[num_items]['RNN_Training_Time'].append(rnn_time)
            results[num_items]['RNN_Loss'].append(rnn_loss)
            results[num_items]['RNN_Memory_Usage'].append(rnn_memory)

            # Train and evaluate DQN
            dqn_accuracy, dqn_time, dqn_loss, dqn_memory = train_and_evaluate_model(dqn_model, data, dqn_model.train_dqn, dqn_model.solve_knapsack_with_dqn, num_items)
            results[num_items]['DQN_Accuracy'].append(dqn_accuracy)
            results[num_items]['DQN_Training_Time'].append(dqn_time)
            results[num_items]['DQN_Loss'].append(dqn_loss)
            results[num_items]['DQN_Memory_Usage'].append(dqn_memory)   

    return results

## Function to save and analyze results
def analyze_results(results):
    for num_items in results.keys():
        print(f"\nResults for item volume: {num_items}")
        for model in ['DNN', 'RNN', 'DQN']:
            avg_accuracy = np.mean(results[num_items][f'{model}_Accuracy'])
            avg_time = np.mean(results[num_items][f'{model}_Training_Time'])
            avg_loss = np.mean([np.mean(loss) for loss in results[num_items][f'{model}_Loss']])
            avg_memory = np.mean(results[num_items][f'{model}_Memory_Usage']) / (1024 * 1024)  # Convert to MB
            print(f"{model} - Average Accuracy: {avg_accuracy:.4f}, Average Training Time: {avg_time:.2f} seconds, "
                  f"Average Loss: {avg_loss:.4f}, Peak Memory Usage: {avg_memory:.2f} MB")

## Usage-----
item_volumes = [5, 10, 20, 50, 100, 200]  # Different item volumes to test
num_samples_per_volume = 1000  # Number of samples per item volume

## Run the automated process
results = automate_process(item_volumes, num_samples_per_volume)

## Analyze and print the results
analyze_results(results)

