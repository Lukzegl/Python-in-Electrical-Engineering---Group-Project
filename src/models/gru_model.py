import torch.nn as nn

class GRUForecaster(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=24, num_layers=2): # [64,24] wymiary
        super(GRUForecaster, self).__init__()
        
        # Warstwa GRU (batch_first=True oznacza, że tensor ma kształt [batch, seq, feature])
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Warstwa liniowa w pełni połączona na końcu
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, hidden = self.gru(x)
        
        # hidden to pamięć z ostatniego kroku czasowego, bo przewidujemy szeregi czasowe
        # Pobieramy stan z ostatniej warstwy GRU -> hidden[-1]
        predictions = self.fc(hidden[-1])
        return predictions