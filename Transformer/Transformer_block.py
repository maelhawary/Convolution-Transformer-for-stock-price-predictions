import torch
import torch.nn as nn
from Transformer import Multi_attention_heads as Matt
import config as config


class TransformerBlock(nn.Module):

    def __init__(self,number_of_heads,d_model,d_att_weigh,seq_len,d_FFN,dropout) -> None:
        super().__init__()
        self.confi=config.get_config()
        self.input_last_shape=self.confi['input_size']
        self.att= Matt.Multiheads( number_of_heads, d_model, d_att_weigh, seq_len, dropout)
        self.LN_1=nn.LayerNorm(self.input_last_shape)
        #self.FFN=FFN.PositionWiseFFN(d_att_weigh, d_FFN,dropout)
        self.dropout=nn.Dropout(dropout)
        self.conv_1=nn.Conv1d(self.confi['seq_len'], self.confi['seq_len'],1)#, stride=1)#, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        
        self.activation=nn.ReLU()
        self.conv_2=nn.Conv1d(self.confi['seq_len'], self.confi['seq_len'],1)#, stride=1)#, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.LN_2=nn.LayerNorm(self.input_last_shape)




    def forward(self,idx):
        #print('think about adding the residual net',idx.shape)
        #out=idx+self.att(self.LN_1(idx))
        #out=out+self.FFN(self.LN_2(out))# (B,T,d_FFN)
        out=self.dropout(self.att(idx))
       # print('dropout',out.shape)
        out=self.LN_1(idx+out)
       # print('LN_1',out.shape)
        out=self.activation(self.conv_1(out))
       # print('conv_1',out.shape)
        out=self.dropout(self.conv_2(out))
       # print('conv_2',out.shape)
        out=self.LN_2(out)
       # print('LN_2',out.shape)

        return out