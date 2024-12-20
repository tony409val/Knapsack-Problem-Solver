import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
class TransformerKnapsackModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TransformerKnapsackModel, self).__init__()

        # Initial normalization and fully connected layers
        self.norm_layer = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 96)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer for Q-values of each action
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, state):
        # Normalize and pass through fully connected layers
        x = self.norm_layer(state)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, num_items, hidden_dim)
        
        # Output layer providing real-valued Q-values for each action
        x = self.output_layer(x).squeeze(1)  # Shape: (batch_size, 2) for 2 actions
        return x

# Test usage

# # Define input dimensions based on updated methodology
# batch_size = 5
# num_items = 7  # Number of items in the knapsack
# input_dim = 6  # 6 features per item (capacity, bag_weight, bag_value, item_weight, item_value, remaining_steps)
# hidden_dim = 96
# num_layers = 2
# num_heads = 4

# # Sample input tensor shaped (batch_size, num_items, input_dim)
# state = torch.rand(batch_size, num_items, input_dim)

# # Initialize model and pass sample data
# model = TransformerKnapsackModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads)

# # Forward pass
# with torch.no_grad():
#     output_probs = model(state)

# # Output results
# print("Item Selection Probabilities (Batch of 5 examples):")
# print(output_probs)
# print("Shape of Output Probabilities:", output_probs.shape)  # Should be (batch_size, num_items)

# # Interpret model predictions as binary selections
# selected_items = (output_probs > 0.5).int()
# print("\nSelected Items (1 = selected, 0 = not selected):")
# print(selected_items)
# print("Shape of Selected Items:", selected_items.shape)  # Should be (batch_size, num_items)