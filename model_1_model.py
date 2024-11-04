import torch
import torch.nn as nn

class MemoryConstructor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MemoryConstructor, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, values, weights, capacities):
        # Concatenate values, weights, capacities

        values = values.unsqueeze(-1)  # Shape: (batch_size, num_items, 1)
        weights = weights.unsqueeze(-1)  # Shape: (batch_size, num_items, 1)
        capacities = capacities.unsqueeze(-1)  # Shape: (batch_size, num_items, 1)
        
        items = torch.cat([values, weights, capacities], dim=-1)  # Shape: (batch_size, num_items, 3)
        
        # Pass through a fully connected layer to get memory representations
        memory = self.fc(items)  # Shape: (batch_size, num_items, hidden_dim)
        return memory

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, memory):
        # memory is the output from the MemoryConstructor (batch_size, num_items, hidden_dim)
        output, hidden = self.gru(memory)  # Shape: (batch_size, num_items, hidden_dim)
        return hidden  # Return the hidden state of the GRU as the encoded representation

class KnapsackHandler(nn.Module):
    def __init__(self, hidden_dim):
        super(KnapsackHandler, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
    
    def forward(self, hidden, current_state):

        # Update the current state of the knapsack
        output, new_state = self.gru(hidden, current_state)  # Update the knapsack state

        return new_state  # Return the new state of the knapsack

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
    
    def forward(self, hidden, memory):
        # Use the last hidden layer
        hidden = hidden[-1] # Shape: (batch_size, hidden_dim)

        # Concatenate hidden state and memory for attention
        hidden_memory = torch.cat((hidden.unsqueeze(1).expand_as(memory), memory), dim=2)
        energy = torch.tanh(self.attn(hidden_memory))  # Shape: (batch_size, num_items, hidden_dim)
        energy = energy.matmul(self.v)  # Shape: (batch_size, num_items)
        attn_weights = torch.softmax(energy, dim=1)  # Shape: (batch_size, num_items)
        return attn_weights

class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super(GRUDecoder, self).__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output layer for probability prediction
    
    def forward(self, hidden, memory, attn_weights):
        # Apply attention to the memory
        context_vector = torch.bmm(attn_weights.unsqueeze(1), memory).squeeze(1)
        output, hidden = self.gru(context_vector.unsqueeze(1), hidden)
        output = torch.sigmoid(self.fc(output)).squeeze(-1)  # Output probabilities
        return output, hidden

# Neural Knapsack Solver: Final Model Combining All Components
class NeuralKnapsackSolver(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralKnapsackSolver, self).__init__()
        self.memory_constructor = MemoryConstructor(input_dim, hidden_dim)
        self.encoder = GRUEncoder(hidden_dim, hidden_dim)
        self.knapsack_handler = KnapsackHandler(hidden_dim)
        self.attention = AttentionMechanism(hidden_dim)
        self.decoder = GRUDecoder(hidden_dim)

    def forward(self, values, weights, capacities):
        # Construct memory
        memory = self.memory_constructor(values, weights, capacities)

        # Encode memory
        hidden = self.encoder(memory)

        # Initialize knapsack state (2 layers, batch_size, hidden_dim)
        batch_size = memory.size(0)
        knapsack_state = torch.zeros(2, batch_size, hidden.size(2), device=hidden.device)

        # Attention and decoding loop
        outputs = []
        for _ in range(memory.size(1)): # Loop over items
            # Compute attention weights
            attn_weights = self.attention(hidden, memory)

            # Use only the last layer of the encoder's hidden state as input to the knapsack handler
            hidden = hidden[-1].unsqueeze(1)  # Shape: (100, 1, 32) to match GRU input format

            # Extract last knapsack state
            knapsack_state = self.knapsack_handler(hidden, knapsack_state)

            decoder_hidden = torch.zeros(3, batch_size, hidden.size(2), device=hidden.device)
            decoder_hidden[:2] = knapsack_state # Assign knapsack_state to the first 2 layers of decoder_hidden

            # Decode to get item decision probabilities
            output, hidden = self.decoder(decoder_hidden, memory, attn_weights)

            outputs.append(output)

        return torch.stack(outputs, dim=1).squeeze(-1) # Remove the last dimension and return probabilities for each item