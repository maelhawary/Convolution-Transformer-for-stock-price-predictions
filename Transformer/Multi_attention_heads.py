import torch
import torch.nn as nn
import config as config

class AttentionHead(nn.Module):

    def __init__(self,d_model,seq_len,head_size,dropout) -> None:
        super().__init__()
        self.confi=config.get_config()

        self.input_size=self.confi['input_size']
        self.K=nn.Linear(self.input_size,head_size, bias=False)
        self.Q=nn.Linear(self.input_size,head_size, bias=False)
        self.V=nn.Linear(self.input_size,head_size, bias=False)        #self.encoding=encoding.Embedding_and_Postional_endocing(vocab_size, d_model,seq_len)
        #self.register_buffer('msk', torch.tril(torch.ones(seq_len, seq_len)))
        self.dropout = nn.Dropout(dropout)
       # nn.init.xavier_uniform(self.K)
       # nn.init.xavier_uniform(self.Q)
      #  nn.init.xavier_uniform(self.V)


    def forward(self,idx):
        #input=self.encoding.embedding(idx)+self.encoding.positioning(idx) #(B.T.d_model)
        B,L,dd = idx.shape ### here in stock it is (B,T,3)
        #print('idx.shape',idx.shape) 
        Key=self.K(idx) ###(B,T,h)
        #print('K_shape',Key.shape)
        Query=self.Q(idx) ###(B,T,h)
        Value=self.V(idx) ###(B,T,h)
        S=Query @ Key.transpose(-2,-1) * Key.shape[-1]**-0.5#(B,L,L)
        S_masked=S#.masked_fill(self.msk[:L, :L] == 0, float('-inf')) # (B, L, L)
        att=nn.functional.softmax(S_masked, dim=-1)
        att = self.dropout(att)# %%% remove this to see the difference
        out= att @ Value ## (B,)
        #return out,att
        return out
    
class Multiheads(nn.Module):
    def __init__(self,number_of_heads,d_model,d_att_weigh,seq_len,dropout) -> None:
        super().__init__()
        self.confi=config.get_config()

        self.input_size=self.confi['input_size']
        assert d_att_weigh % number_of_heads == 0, "d_att_weigh is not divisible by number_of_heads"
        self.head_size=d_att_weigh // number_of_heads
        self.Multiheads=nn.ModuleList([AttentionHead(d_model,seq_len,self.head_size,dropout) for i in range (number_of_heads) ])#
        self.proj = nn.Linear(self.head_size * number_of_heads, self.input_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.Multiheads], dim=-1)
        #print('there is no drpout in the original')
       # print('cat',out.shape)

        out = self.dropout(self.proj(out))
        #print('proj',out.shape)

        return out
