import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from PIL import Image, ImageTk
import math
import matplotlib.pyplot as plt
import os

# Neural Network Models Architecture Classes
class KnapsackDNN(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super(KnapsackDNN, self).__init__()
        self.layers = nn.ModuleList()

        #Input Layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        #Hidden Layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        #Output Layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        
        x = self.sigmoid(self.layers[-1](x))
        return x
        
class KnapsackRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_layers, dropout_prob=0.7):
        super(KnapsackRNN, self).__init__()
        self.hidden_size = hidden_sizes[0]
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=2, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.hidden_size + 1, hidden_sizes[1])])
        for i in range(1, len(hidden_sizes) - 1):
            self.fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size)
        
        # Dropout layer after LSTM output to reduce overfitting
        self.dropout = nn.Dropout(dropout_prob)
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, capacity):
    
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]

        #capacity
        expanded_capacity = capacity.view(-1, 1).expand(x.size(0), 1)
        out = torch.cat((out, expanded_capacity), dim=1)
        
        for fc in self.fc_layers:
            out = self.relu(fc(out))
        out = self.sigmoid(self.fc_out(out))
        
        return out

class KnapsackDQN(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super(KnapsackDQN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=0.2)

        #Input Layer
        self.layers.append(nn.Linear(input_dim, hidden_sizes[0]))

        #Hidden Layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        #Hidden Layers
        self.layers.append(nn.Linear(hidden_sizes[-1], output_dim))


        self.relu = nn.ReLU()
        

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            x = self.dropout(x) # Apply dropout
        x = self.layers[-1](x)
        return x

# Knapsack Environment Class for the Deep Q-Learning Model
class KnapsackEnv:
    def __init__(self, items, capacity, 
                 reward_for_valid_selection= 2.1, 
                 penalty_for_exceeding_capacity= -4.0, 
                 bonus_for_full_capacity_use= 0.45, 
                 penalty_for_unused_capacity= 0.25):
        self.items = items
        self.capacity = capacity
        self.state = None
        self.current_item = 0

        self.reward_for_valid_selection = reward_for_valid_selection
        self.penalty_for_exceeding_capacity = penalty_for_exceeding_capacity
        self.bonus_for_full_capacity_use = bonus_for_full_capacity_use
        self.penalty_for_unused_capacity = penalty_for_unused_capacity

        self.reset()

    def reset(self):
        self.state = (0, self.capacity, self.items[0][0], self.items[0][1])  # Start with no items and full capacity, and first item details
        self.current_item = 0
        return self.state
    
    def step(self, action):
        current_value, remaining_capacity, _, _ = self.state


        if action == 1 and remaining_capacity >= self.items[self.current_item][1]:

            # Reward for selecting a valid item + additional reward for value-weight efficiency
            reward = self.items[self.current_item][0] * self.reward_for_valid_selection

            next_value = current_value + self.items[self.current_item][0]
            remaining_capacity -= self.items[self.current_item][1]
            
        elif action == 1:
            reward = self.penalty_for_exceeding_capacity # Penalty for trying to select an item that exceeds the capacity
            next_value = current_value
        else:
            reward = 0 # No reward for not selecting item
            next_value = current_value

        self.current_item += 1
        done = self.current_item == len(self.items)

        if done:
            # Reward based on how well the capacity is used
            if remaining_capacity == 0: 
                reward += next_value * self.bonus_for_full_capacity_use
            else:
                next_value * self.penalty_for_unused_capacity

        # Normalize the reward
        reward /= 10.0

        if not done:
            next_state = (next_value, remaining_capacity, self.items[self.current_item][0], self.items[self.current_item][1])
        else:
            next_state = (next_value, remaining_capacity, 0, 0) # No more items to consider
        self.state = next_state
        return self.state, reward, done

    def action_space(self):
        return [0, 1]

    def observation_space(self):
        return 4  # Value, remaining capacity, current item value, current item weight     

# Experience Replay Class for the Deep Q-learning Model

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Deep Q-learning Model Agent class
class DQNAgent:
    def __init__(self, input_dim, hidden_sizes, output_dim,
                 lr=  0.0001, 
                 batch_size = 128,
                 gamma = 0.99,
                 target_update = 10,
                 decay_rate = 0.99,
                 weight_decay = 1e-4,
                 memory_size = 10000,
                 epsilon_start = 1.0,
                 epsilon_end = 0.01):
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.decay_rate = decay_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.policy_net = KnapsackDQN(input_dim, hidden_sizes, output_dim)
        self.target_net = KnapsackDQN(input_dim, hidden_sizes, output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.SmoothL1Loss()


    def select_action(self, state, eps_threshold):
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(torch.tensor(state, dtype=torch.float32)).argmax().item()
        else:
            return random.choice([0, 1])
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch.next_state if s is not None])
        state_batch = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch.state])
        action_batch = torch.tensor(batch.action, dtype=torch.float32).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.long())
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def train_dqn(self, data, visualization, num_episodes=1000, progress_bar=None):
        # Epsilon decay parameters
        epsilon_start = self.epsilon_start
        epsilon_end = self.epsilon_end
        decay_rate = self.decay_rate
        epsilon = epsilon_start
        

        loss_values = [] # For loss plot visualization

        # Setup loss plot
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(0, num_episodes)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.title('Training Loss')

        # Get the matplotlib figure manager
        plot_manager = plt.get_current_fig_manager()

        for episode in range(num_episodes):
            # Periodically boost exploration
            if episode % 50 == 0 and episode != 0:
                epsilon = min(epsilon + 0.1, 1.0) # Increase epsilon periodically to reintroduce exploration

            # Select a random sample from the generated data
            sample = random.choice(data)
            items = [(v, w) for v, w in zip(sample[0], sample[1])]
            capacity = sample[2]
            env = KnapsackEnv(items, capacity)
            
            state = env.reset()
            episode_loss = 0.0
            actions_taken = [] # List to save selected items for visualization

            for t in range(len(items)):
                eps_threshold = max(epsilon_end, epsilon)
                action = self.select_action(state, eps_threshold)

                actions_taken.append(action)

                next_state, reward, done = env.step(action)

                self.memory.push(state, action, next_state if not done else None, reward)
                state = next_state
                loss = self.optimize_model()
                episode_loss += loss
                if done:
                    break

            loss_values.append(episode_loss / (t + 1)) # Storage average loss for this episode
           
            # Update the target network periodically
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon after each episode
            epsilon *= decay_rate
            epsilon = max(epsilon, epsilon_end)

            # Update visualization after each episode
            values = [item[0] for item in items]
            weights = [item[1] for item in items]
            prediction = actions_taken
            visualization.update(prediction, values, weights)

            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Loss: {episode_loss / (t + 1)}") 
                if progress_bar:
                    progress_bar["value"] = (episode + 1) * 100 / num_episodes
                    progress_bar.update()  

            # Update loss plot
            line.set_xdata(range(1, episode + 2))
            line.set_ydata(loss_values)
            ax.relim()
            ax.autoscale_view()
            plt.draw()

            # Position the plot window next to tkinter window
            position_plot_window(visualization, plot_manager)

        plt.ioff()
        plt.show


        if progress_bar:
            progress_bar["value"] = 100
            progress_bar.update()
    
    # Deep-Q Learning Model Prediction Function 
    def solve_knapsack_with_dqn(self, values, weights, knapsack_capacity):
        items = [(v, w) for v, w in zip(values, weights)]
        env = KnapsackEnv(items, knapsack_capacity)
        state = env.reset()
        selected_items = []
        prediction = []
        total_value = 0
        total_weight = 0

        for i in range(len(items)):
            with torch.no_grad():
                q_values = self.policy_net(torch.tensor(state, dtype=torch.float32))
            action = q_values.argmax().item()


            selected_items.append(action)
            next_state, reward, done = env.step(action)
            state = next_state

            #Update total value and weight 
            if action == 1:
                total_value += items[i][0]
                total_weight += items[i][1]

            if done:
                break
        
        prediction.append((selected_items, total_value, total_weight))

        return prediction

# Vizualization Class
class KnapsackVisualization:

    def __init__(self, root, num_items, data, knapsack_capacity):
        self.root = root
        self.num_items = num_items
        self.knapsack_capacity = knapsack_capacity

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Maximum percentage of screen size to be used
        max_width_percentage = 0.5
        max_height_percentage = 0.5

        # Calculate the maximum allowable canvas size
        max_canvas_width = int(screen_width * max_width_percentage)
        max_canvas_height = int(screen_height * max_height_percentage)

        # Margin around the edges
        margin = 50

        # Adjust canvas size based on the number of items
        self.grid_size = math.ceil(math.sqrt(num_items)) # create a near-square grid

        # Size of each cell in the grid
        initial_cell_size = 100 

       # Calculate required canvas size
        required_width = self.grid_size * initial_cell_size + margin * 2 + 200 # Add extra width for the knapsack
        required_height = self.grid_size * initial_cell_size + margin * 2

        # Determine scaling factor based on available screen size
        scaling_factor = min(max_canvas_width / required_width, max_canvas_height / required_height)

        # Apply scaling factor to cell size and canvas size
        self.cell_size = int(initial_cell_size * scaling_factor)
        self.canvas_width = int(required_width * scaling_factor)
        self.canvas_height = int(required_height * scaling_factor)
        self.knapsack_width = int(150 * scaling_factor) # Scale knapsack size
        self.knapsack_height = int(150 * scaling_factor)

        self.knapsack_center_x = self.canvas_width - self.knapsack_width / 2 - margin
        self.knapsack_center_y = self.knapsack_height / 2

        # Create the canvas
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        # Load the knapsack image
        base_path = os.path.dirname(os.path.abspath(__file__)) # Get the current directory of the script
        self.knapsack_image_path = os.path.join(base_path, "icons", "knapsack.png")
        self.knapsack_image = ImageTk.PhotoImage(Image.open(self.knapsack_image_path).resize((self.knapsack_width, self.knapsack_height)))
        self.canvas.create_image(self.knapsack_center_x, self.knapsack_center_y, image=self.knapsack_image, anchor=tk.CENTER)

        # Initialize the item image paths list with the same path for each item
        item_image_path = os.path.join(base_path, "icons", "circle.png")
        selected_image_path = os.path.join(base_path, "icons", "circle_green.png")
        
        # Load different state images and scale them
        self.original_image = ImageTk.PhotoImage(Image.open(item_image_path).resize((int(50 * scaling_factor), int(50 * scaling_factor))))
        self.selected_image = ImageTk.PhotoImage(Image.open(selected_image_path).resize((int(50 * scaling_factor), int(50 * scaling_factor))))    

        # Initialize text for displaying total value and weight
        self.total_value_text = None
        self.total_weight_text = None   

    def draw_grid(self):
        self.canvas.delete("item")
        self.item_labels = []
        self.value_labels = []
        self.weight_labels = []

        for i in range(self.num_items):
            value = self.item_values[i]
            weight = self.item_weights[i]

            # Calculate row and column
            row = i // self.grid_size
            col = i % self.grid_size

            # Calculate position based on the row and column
            x_position = 50 + col * self.cell_size
            y_position = 50 + row * self.cell_size
            label = self.canvas.create_image(x_position, y_position, image=self.original_image, anchor=tk.CENTER, tags="item")
            self.item_labels.append(label)

            # Draw the value and weight under each icon
            value_label = self.canvas.create_text(x_position, y_position + 30, text=f"V: {value}", tags="item")
            weight_label = self.canvas.create_text(x_position, y_position + 50, text=f"W: {weight}", tags="item")
            self.value_labels.append(value_label)
            self.weight_labels.append(weight_label)
    
    def update(self, selected_items, values, weights):
        # Update item values and weights for the current sample
        self.item_values = values
        self.item_weights = weights

        self.draw_grid() # Clear and redraw grid with each update
        
        # Calculate total value and weight of selected items
        total_value = sum(v for i, v in enumerate(self.item_values) if selected_items[i])
        total_weight = sum(w for i, w in enumerate(self.item_weights) if selected_items[i])

        # Calculate the position for the labels based on the knapsack image size
        label_y_offset = self.knapsack_center_y + self.knapsack_height / 2 + 20

        # Display total value and weight below the knapsack image
        if self.total_value_text is None:
            self.total_value_text = self.canvas.create_text(
            self.knapsack_center_x, label_y_offset,
            text=f"Total Value: {total_value}", fill="black", font=("Arial", 14)
        )
        else:
            self.canvas.itemconfig(self.total_value_text, text=f"Total Value: {total_value}")

        if self.total_weight_text is None:    
            self.total_weight_text = self.canvas.create_text(
            self.knapsack_center_x, label_y_offset + 20,
            text=f"Total Weight: {total_weight}", fill="black", font=("Arial", 14)
        )
        else:
            self.canvas.itemconfig(self.total_weight_text, text=f"Total Weight: {total_weight}")

        self.root.update()

        def change_color_and_move(i):
            if selected_items[i]:
                # Change to selected image
                self.canvas.itemconfig(self.item_labels[i], image=self.selected_image)
                self.canvas.itemconfig(self.value_labels[i], fill="green")
                self.canvas.itemconfig(self.weight_labels[i], fill="green")
                self.canvas.tag_raise(self.item_labels[i])
                self.root.update_idletasks()

                # Move the icon to the knapsack area
                new_x = self.knapsack_center_x + random.randint(-50,50)
                new_y = self.knapsack_center_y + random.randint(-50, 50)
                self.canvas.coords(self.item_labels[i], new_x, new_y)
                self.root.update_idletasks()            


        # Schedule changes for each item
        for i in range(len(selected_items)):
            change_color_and_move(i)

# Loss Plot Window Position Manager
def position_plot_window(visualization, plot_manager):
    # Ensure the Tkinter window has been fully rendered
    visualization.root.update_idletasks()

    # Get the current geometry of the Tkinter window
    tk_x = visualization.root.winfo_x()
    tk_y = visualization.root.winfo_y()
    tk_width = visualization.root.winfo_width()

    # Calculate position for the Matplotlib window
    plot_x = tk_x + tk_width + 20  # Place it 20 pixels to the right of the Tkinter window
    plot_y = tk_y  # Align the top edges

    # Set the Matplotlib window position
    plot_manager.window.wm_geometry(f"+{plot_x}+{plot_y}")

# Data Generation function
def generate_data(num_samples, num_items):
    data = []
    for _ in range(num_samples):
        values = [random.randint(1, 50) for _ in range(num_items)]
        weights = [random.randint(1, 20) for _ in range(num_items)]
        knapsack_capacity = 50
        solution = knapsack_dp(weights, values, knapsack_capacity)
        data.append((values, weights, knapsack_capacity, solution[0][0]))
    return data

# Optimal Solution using Dynamic Programming
def knapsack_dp(weights, values, capacity):
    """Solve the knapsack problem by dynamic programming."""
    n = len(weights)
    dp = np.zeros((n + 1, capacity + 1))
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    # Reconstruct the solution
    solution = []
    selected_items = np.zeros(n, dtype=int)
    optimal_value = 0
    optimal_weight = 0
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items[i-1] = 1
            w -= weights[i-1]

            optimal_value += values[i-1]
            optimal_weight += weights[i-1]

    solution.append((selected_items, optimal_value, optimal_weight))

    return solution

# Data Preparation function for the RNN Model
def prepare_rnn_data(training_data):
    input_tensors = []
    label_tensors = []
    capacity = []
    
    # Combine values and weights into a single input sequence and convert to a tensor
    for item in training_data:
        values, weights, knapsack_capacity, selected_items   = item
       
        # Each input sequence is treated as a batch of size 1 
        input_sequence = torch.tensor([[v, w] for v, w in zip(values, weights)], dtype=torch.float32).unsqueeze(0)

        input_tensors.append(input_sequence)
        label_tensors.append(torch.tensor(selected_items, dtype=torch.float32).unsqueeze(0))
        capacity.append(knapsack_capacity)
    
    return list(zip(input_tensors, label_tensors, capacity))

# Deep Neural Network Model Training Function
def train_nn(model, data, visualization, epochs=1000, lr=0.01, progress_bar=None):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_values = [] # For loss plot visualization

    # Setup loss plot
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, epochs)
    ax.set_ylim(0, 0.01)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.title('Training Loss')

    # Get the matplotlib figure manager
    plot_manager = plt.get_current_fig_manager()

    # Initialize first sample for visualization
    sample_index = 0
    current_sample = data[sample_index]

    for epoch in range(epochs):
        total_loss = 0.0
    
        for values, weights, knapsack_capacity, selected_items   in data:
            input_data = torch.FloatTensor(values + weights + [knapsack_capacity])
            target = torch.FloatTensor(selected_items)

            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss_values.append(total_loss / len(data)) # Storage average loss for this epoch

        # Change visualization sample every 5 epochs
        if (epoch + 1) % 5 == 0:
            sample_index = (sample_index + 1) % len(data) # Cycle through data samples
            current_sample = data[sample_index]

        # Get the values, weights, and selected items for the current sample
        current_values, current_weights, current_capacity, _ = current_sample

        # Recompute prediction for the current sample
        input_data = torch.FloatTensor(current_values + current_weights + [current_capacity])
        prediction_output = model(input_data)
        prediction = [1 if prob > 0.5 else 0 for prob in prediction_output.detach().numpy()]

        # Update visualization  
        visualization.update(prediction, current_values, current_weights)   
 
        # Update progress bar and print average loss of this epoch
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data)}")
            if progress_bar:
                progress_bar["value"] = (epoch + 1) * 100 / epochs
                progress_bar.update()

        # Update loss plot
        line.set_xdata(range(1, epoch + 2))
        line.set_ydata(loss_values)
        ax.relim()
        ax.autoscale_view()
        plt.draw()

        # Position the plot window next to tkinter window
        position_plot_window(visualization, plot_manager)

    plt.ioff()
    plt.show

    if progress_bar:
        progress_bar["value"] = 100
        progress_bar.update()
  
# Recurrent Neural Network Model Training Function
def train_rnn(model, data, visualization, epochs=1000, lr=0.01, progress_bar=None):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_values = [] # For loss plot visualization

    #Setup loss plot
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, epochs)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.title('Training Loss')

    #Get the matplotlib figure manager
    plot_manager = plt.get_current_fig_manager()

    # Initialize first sample for visualization
    sample_index = 0
    current_sample = data[sample_index]

    for epoch in range(epochs):
        total_loss = 0.0
        for input_tensors, label_tensors, capacity in data:
            
            optimizer.zero_grad()

            capacity_tensor = torch.tensor([capacity], dtype=torch.float32).unsqueeze(0).to(input_tensors.device)
        
            output = model(input_tensors, capacity_tensor)
            loss = criterion(output, label_tensors)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        loss_values.append(total_loss / len(data)) # Storage average loss for this epoch

        # Change visualization sample every 5 epochs
        if (epoch + 1) % 5 == 0:
            sample_index = (sample_index + 1) % len(data) # Cycle through data samples
            current_sample = data[sample_index]

        # Get the values, weights, and selected items for the current sample    
        input_tensors, _, capacity = current_sample
        values = input_tensors.squeeze(0).detach().numpy()[:, 0].tolist() # Extract values
        weights = input_tensors.squeeze(0).detach().numpy()[:, 1].tolist() # Extract weights

        # Recompute prediction for the current sample
        prediction_output = model(input_tensors, capacity_tensor)
        prediction = [1 if prob > 0.5 else 0 for prob in prediction_output.squeeze().detach().numpy()] # Predicted selection
            
        # Update visualization
        visualization.update(prediction, values, weights)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data)}")
            if progress_bar:
                progress_bar["value"] = (epoch + 1) * 100 / epochs
                progress_bar.update()

        # Update loss plot
        line.set_xdata(range(1, epoch + 2))
        line.set_ydata(loss_values)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

        # Position the plot window next to tkinter window
        position_plot_window(visualization, plot_manager)

    plt.ioff()
    plt.show

    if progress_bar:
        progress_bar["value"] = 100
        progress_bar.update()
    
# Solution of the DNN Trained Model
def solve_knapsack_with_nn(model, values, weights, knapsack_capacity):
    input_data = torch.FloatTensor(values + weights + [knapsack_capacity])
    prediction = []
    

    output = model(input_data)

    selected_items = [1 if prob > 0.5 else 0 for prob in output]
    total_value = sum(v for v, selected in zip(values, selected_items) if selected)
    total_weight = sum(w for w, selected in zip(weights, selected_items) if selected)

    prediction.append((selected_items, total_value, total_weight))

    return prediction

# Solution of the RNN Trained Model
def solve_knapsack_with_rnn(model, values, weights, knapsack_capacity):
    prediction = []
    # Prepare the input sequence with 2 features per timestep
    input_features = [[values[i], weights[i]] for i in range(len(values))]
    input_tensor = torch.FloatTensor(input_features).unsqueeze(0)  # Adding the batch dimension
    capacity_tensor = torch.tensor([knapsack_capacity], dtype=torch.float32).unsqueeze(0).to(input_tensor.device)

    # Pass the reshaped input_data to the model
    output = model(input_tensor, capacity_tensor)
  
    selected_items = [1 if prob > 0.5 else 0 for prob in output.squeeze()]  #squeeze for correct output format
    total_value = sum(v for v, selected in zip(values, selected_items) if selected)
    total_weight = sum(w for w, selected in zip(weights, selected_items) if selected)

    prediction.append((selected_items, total_value, total_weight))

    return prediction

# Main function
def solve_knapsack(root, num_items, algorithm_var, progress_bar):
    selected_algorithm = algorithm_var.get()

    training_data = generate_data(1000, num_items)

    single_point_data = generate_data(100, num_items) # smaller sample to debug / present visualization

    dnn_model = KnapsackDNN(2 * num_items + 1, [256, 128], num_items)
    dqn_model = DQNAgent(input_dim=4, hidden_sizes=[256, 128], output_dim=2)
    rnn_model = KnapsackRNN(2 * num_items + 1, [64, 64], num_items, num_layers=1)  
    
    visualization = KnapsackVisualization(root, num_items, training_data, 50)

    # Global variables to hold the trained model and algorithm
    global trained_model, current_algorithm
    trained_model = None
    current_algorithm = selected_algorithm

    if selected_algorithm == "Deep Neural Network":
        
        train_nn(dnn_model, single_point_data, visualization, progress_bar=progress_bar)
        trained_model = dnn_model

    elif selected_algorithm == "Deep-Q Learning":  
        
        dqn_model.train_dqn(single_point_data, visualization, progress_bar=progress_bar)
        trained_model = dqn_model


    elif selected_algorithm == "Recurrent Neural Network":

        prepared_data = prepare_rnn_data(single_point_data)  

        train_rnn(rnn_model, prepared_data, visualization, progress_bar=progress_bar)
        trained_model = rnn_model

    
    else:
        messagebox.showerror("Error", "Wrong choice of algorithm.")
        return
    
    messagebox.showinfo("Training Complete", "The model has been trained. You can now test a new sample.")

# Give New Values to the Trained Model and Show Results
def test_new_sample(item_var):
    global trained_model, current_algorithm

    if not trained_model:
        messagebox.showerror("Error", "No model has been trained yet!")
        return
    
    num_items = int(item_var.get()) # Get the number of items from the GUI
    test_values = [random.randint(1, 50) for _ in range(num_items)]
    test_weights = [random.randint(1, 20) for _ in range(num_items)]
    test_knapsack_capacity = 50

    if current_algorithm == "Deep Neural Network":
        prediction = solve_knapsack_with_nn(trained_model, test_values, test_weights, test_knapsack_capacity)

    elif current_algorithm == "Deep-Q Learning":
        prediction = trained_model.solve_knapsack_with_dqn(test_values, test_weights, test_knapsack_capacity)

    elif current_algorithm == "Recurrent Neural Network":
        prediction = solve_knapsack_with_rnn(trained_model, test_values, test_weights, test_knapsack_capacity)
    
    else:
        messagebox.showerror("Error", "Unknown algorithm type!")
        return

    solution = knapsack_dp(test_weights, test_values, test_knapsack_capacity)

    result_message = (f"Test Sample Values: {test_values}\n"
                      f"Test Sample Weights: {test_weights}\n"
                      f"Predicted Items: {prediction[0][0]}\n"
                      f"Total Value: {prediction[0][1]}\n"
                      f"Total Weight: {prediction[0][2]}\n"
                      f"Optimal Item Selection: {solution[0][0]}\n"
                      f"Optimal Total Value: {solution[0][1]}\n"
                      f"Optimal Total Weight: {solution[0][2]}")

    messagebox.showinfo("Results", result_message)

# GUI Function
def create_gui():
    root = tk.Tk()
    root.title("0/1 Knapsack Problem Solver")

    # Training start function
    def start_solving():
        num_items = int(item_var.get())
        solve_knapsack(root, num_items, algorithm_var, progress_bar)

    # Item Volume Selection Label
    item_label = ttk.Label(root, text="Select Item Volume:")
    item_label.pack(pady=10)

    item_var = tk.StringVar()
    item_var.set("10")  # Default value
    item_dropdown = ttk.Combobox(root, textvariable=item_var, values=["5", "10", "20", "50", "100", "200"])
    item_dropdown.pack(pady=10)

    # Algorithm Selection Label
    algorithm_label = ttk.Label(root, text="Select Model:")
    algorithm_label.pack(pady=10)

    algorithm_var = tk.StringVar()
    algorithm_var.set("Deep Neural Network")  # default value
    algorithm_dropdown = ttk.Combobox(root, textvariable=algorithm_var, values=["Deep Neural Network", "Deep-Q Learning", "Recurrent Neural Network"])
    algorithm_dropdown.pack(pady=10)

    # Training Start Button
    solve_button = ttk.Button(root, text="Start Training", command=start_solving)
    solve_button.pack(pady=20)

    # Button for testing a new sample
    test_button = ttk.Button(root, text="Test New Sample", command=lambda: test_new_sample(item_var))
    test_button.pack(pady=20)

    # Training progress bar
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
    progress_bar.pack(pady=10)

    # Tkinter loop start
    root.mainloop()
    print("Application Closed")

# GUI Call
create_gui()

