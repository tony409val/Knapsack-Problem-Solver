
class KnapsackEnv:
    def __init__(self, values, weights, capacities):
        self.all_values = values  # Array of item values
        self.all_weights = weights  # Array of item weights
        self.all_capacities = capacities  # capacities
        self.num_instances = len(values) # total number of instances
        
        # Initialize tracking for the current instance
        self.instance_index = 0
        self.num_items = len(self.all_values[0]) # All instances have the same number of items

        self.state = None  # Tracks selected items and remaining capacities
        self.bag_weight = 0
        self.bag_value = 0
        self.remaining_steps = self.num_items
        self.current_index = 0

    def reset(self):
        """Resets the environment for a new episode."""
        # Reset to a new instance using instance_index
        self.values = self.all_values[self.instance_index]
        self.weights = self.all_weights[self.instance_index]
        self.capacity = self.all_capacities[self.instance_index]

        self.state = [0] * self.num_items  # No items selected initially
        self.bag_weight = 0
        self.bag_value = 0
        self.remaining_steps = self.num_items
        self.current_index = 0

        # Increment instance_index to move to the next instance on the next reset
        self.instance_index = (self.instance_index + 1) % self.num_instances # Cycle through instances
        return self._get_observation()

    def step(self, action):
        """Takes an action: 0 (skip item) or 1 (select item)."""
        reward = 0
        if action == 1 and self.weights[self.current_index] <= self.capacity[self.current_index] - self.bag_weight:
            # Select item if within capacities
            self.state[self.current_index] = 1
            self.bag_weight += self.weights[self.current_index]
            self.bag_value += self.values[self.current_index]
            reward = self.values[self.current_index]  # Reward for adding item

        elif action == 0:
            reward = -20  # Negative reward for skipping item

        # Move to the next item
        self.current_index += 1
        self.remaining_steps -= 1
        done = self.current_index >= self.num_items
        return self._get_observation(), reward, done
    

    def _get_observation(self):
        """Returns the current state: values, weights, capacities, and selection state."""
        return {
            "capacity": self.capacity,
            "bag_weight": self.bag_weight,
            "bag_value": self.bag_value,
            "item_weight": self.weights[self.current_index] if self.current_index < self.num_items else 0,
            "item_value": self.values[self.current_index] if self.current_index < self.num_items else 0,
            "remaining_steps": self.remaining_steps
        }

    def get_current_index(self):
        """Returns the current instance index being processed"""
        return self.instance_index

    def render(self):
        """Renders the current state for debugging purposes."""
        print(f"Selected items: {self.state}")
        print(f"Total value: {self.bag_value}, Remaining capacities: {self.capacities - self.bag_weight}")


# Test Usage

# # Sample values and weights
# values = [10, 5, 15, 7, 6, 18, 3]    # Example values for items
# weights = [2, 3, 5, 7, 1, 4, 1]      # Corresponding weights for items
# capacities = 15                        # Knapsack capacities

# # Initialize the knapsack environment
# env = KnapsackEnv(values, weights, capacities)

# # Reset the environment for a new episode
# obs = env.reset()
# print("Initial observation:", obs)

# # Run through an episode with random actions
# done = False
# total_reward = 0

# while not done:
#     # Take a random action (0 = skip item, 1 = add item)
#     action = np.random.choice([0, 1])
    
#     # Step in the environment with the chosen action
#     obs, reward, done = env.step(action)
    
#     # Accumulate reward
#     total_reward += reward
    
#     # Print out the details after each step
#     print(f"Action taken: {action}")
#     print(f"Observation: {obs}")
#     print(f"Reward received: {reward}")
#     print(f"Done: {done}")
#     print("---")

# # Final results after the episode ends
# print("Final total reward:", total_reward)
# env.render()