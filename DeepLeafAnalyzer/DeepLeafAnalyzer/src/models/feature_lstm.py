import torch.nn as nn


class FeatureSequenceLSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=256):
        super().__init__()
        self.lstm=nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
    def forward(self,x):
        # Spatial positions are ordered feature-sequence positions, not physical time.
        seq=x.flatten(2).transpose(1,2)
        _,(h,_) = self.lstm(seq)
        return h[-1]
