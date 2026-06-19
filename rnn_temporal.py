# models/rnn_temporal.py
import torch
import torch.nn as nn
from config import SystemConfig

class TemporalRNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=SystemConfig.RNN_HIDDEN_SIZE, 
                            num_layers=2, 
                            batch_first=True)
        self.mlp = nn.Linear(SystemConfig.RNN_HIDDEN_SIZE, SystemConfig.NUM_COGNITIVE_STATES)

    def forward(self, sequence_embeddings):
        # h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
        rnn_out, _ = self.lstm(sequence_embeddings)
        final_state = rnn_out[:, -1, :] # Many-to-One
        return self.mlp(final_state)