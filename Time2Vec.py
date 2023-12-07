import torch
import torch.nn as nn
import config as config

class Time2VecPositonalEncoding(nn.Module):
    def __init__( self, activation="sine"):
        super().__init__()        
        self.confi=config.get_config()
        self.linear_layer = nn.Linear(self.confi['seq_len'], self.confi['seq_len'])
        self.periodic_layer = nn.Linear(self.confi['seq_len'], self.confi['seq_len'])
        self.activation = activation

    def forward(self, x):
       # print('in vec2',x.shape)
        x=x
        if self.activation == "sine":
            periodic_out = torch.sin(self.periodic_layer(x))
        elif self.activation == "cos":
            periodic_out = torch.cos(self.periodic_layer(x))
        periodic_out=periodic_out.unsqueeze(-1)
     
        original_out = self.linear_layer(x)
        original_out=original_out.unsqueeze(-1)

        out = torch.cat([periodic_out, original_out], 2) ## dimension (B,T,2)
       # print('time2vec_shape',out.shape)
        return out
    

