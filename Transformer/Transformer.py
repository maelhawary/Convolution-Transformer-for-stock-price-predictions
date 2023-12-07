import torch
import torch.nn as nn
from Transformer import Transformer_block as Trans
import config as config
from Time2Vec import Time2VecPositonalEncoding as time_pos 


class DecoderBlock(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.confi=config.get_config()
        self.d_model=self.confi['d_model']
        self.seq_len= self.confi['seq_len']
        self.number_of_heads=self.confi['number_of_heads']
        self.number_of_layers=self.confi['number_of_layers']
        self.d_att_weigh=self.confi['d_att_weigh']
        self.d_FFN=self.confi['d_FFN']
        self.dropout=self.confi['dropout']
        self.t2v=time_pos()
        #self.encod=encoding.Embedding_and_Postional_endocing(vocab_size, self.d_model,self.seq_len)
        #self.embd=embd.InputEmbeddings(self.d_model)   
        #self.pos=position.PositionalEncoding(self.d_model,self.seq_len,self.dropout)

        self.blocks= nn.Sequential(*[Trans.TransformerBlock(self.number_of_heads,self.d_model,self.d_att_weigh,
                                                            self.seq_len,self.d_FFN,self.dropout) for i in range(self.number_of_layers)])
        self.LN=nn.LayerNorm(self.d_model)
        #self.output = nn.Linear(self.d_model, vocab_size)
        self.device=device
        self.average_pool=nn.AdaptiveAvgPool1d(1)
        self.ffw=nn.Linear(self.confi['seq_len'],64)
        self.out=nn.Linear(64, 1)
        self.activ=nn.ReLU()
    def forward(self,idx,ground_truth=None):
        #print('beforevec2',idx.shape)

        B,T=idx.shape
        #x=self.embd(idx)+self.pos(torch.arange(T, device=self.device))
        x=self.t2v(idx) #shape (B,T,3)
        #print('aftert2vec',x.shape)
        idx=idx.unsqueeze(-1)
        x= torch.cat([ x,idx],axis=-1)
        #print('cat',x.shape)
        ##print('shape_after_embd',x.shape)
        #x=self.pos(x)
        x=self.blocks(x)
        #print('blocks',x.shape) ##(B,T,3)

        x = self.average_pool(x)
        x=x.squeeze(-1)
        #print('average_pool',x.shape) ##(B,T,3)

        x =nn.Dropout(self.dropout)(x)
        x =self.ffw(x)
        x=self.activ(x)    
       # print('ffw',x.shape) ##(B,T,3)
        x = nn.Dropout(self.dropout)(x)
        out= self.out(x)

       # print('out',out.shape) ##(B,1)



        if ground_truth is None:
            loss = None
        else:
            loss=torch.mean(torch.square(out-ground_truth))
           # print('lossss',loss) 

        return out, loss



