import torch
import torch.nn as nn
class SimpleLSTM(nn.Module):
    def __init__(self, input_size = 8, output_size = 1, hidden_size=100, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True) 
        self.fc = nn.Linear(self.hidden_size, self.output_size) 

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, cell
    
    def forward(self, x):
        # hidden, cell state init
        h, c = self.init_hidden(x.size(0))
        h, c = h.to(x.device), c.to(x.device)
        out, (h, c) = self.lstm(x, (h, c))     
        final_output = self.fc(out[:, -1:, :])     
        final_output = torch.squeeze(final_output, dim = 1) # shape (100,1)

        return final_output