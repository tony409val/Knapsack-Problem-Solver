import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralKnapsackSolver(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralKnapsackSolver, self).__init__()

        # Memory Constructor: GRU-based Network
        self.memory_constructor = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)

        # Encorder: GRU-based network to encode the entire knapsack instance
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)

        # Knapsack Handler: GRU-based network to maintain the state of the knapsack
        self.knapsack_handler = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)

        # Attention mechanism over memory
        self.attention = nn.Linear(hidden_dim, hidden_dim)

        # Decoder: GRU-based Network to generate output
        self.decoder = nn.GRU(input_dim + hidden_dim, hidden_dim, num_layers=3, batch_first=True)

        # Final output layer to get probabilities (0 or 1 for each item)
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x is the input tensor containing item values, weights, and capacity
        # x.shape = (batch_size, seq_length, input_dim)

        # Memory construction
        memory_output, _ = self.memory_constructor(x)

        # Encoding the knapsack instance
        encoder_output, _ = self.encoder(x)

        # Initialize the knapsack state with zeros
        knapsack_state = torch.zeros(x.size(0), memory_output.size(2), device=x.device)

        # List to store outputs
        outputs = []

        # Iterate through the sequence
        for t in range(x.size(1)):

            # Calculate attention weights
            attn_scores = torch.matmul(memory_output, knapsack_state.unsqueeze(2)).squeeze(2)
            attn_weights = F.softmax(attn_scores, dim=1)

            # Correct the shape for attention weights to multiply with memory_output
            context_vector = torch.sum(memory_output * attn_weights.unsqueeze(2), dim=1)
            
            # Prepare the input for the decoder
            decoder_input = torch.cat((x[:, t, :], context_vector), dim=1).unsqueeze(1)
            
            # Decode the current step
            decoder_output, _ = self.decoder(decoder_input)
            
            # Update the knapsack handler (knapsack state)
            knapsack_state, _ = self.knapsack_handler(decoder_output)
            knapsack_state = knapsack_state.squeeze(1)
            
            # Get the output probability for the current item
            output = torch.sigmoid(self.output_layer(knapsack_state))
            outputs.append(output)

        # Concatenate all outputs along the sequence dimension
        outputs = torch.cat(outputs, dim=1)

        # Return the outputs
        return outputs
    