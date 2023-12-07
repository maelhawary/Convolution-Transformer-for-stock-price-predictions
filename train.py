import Dataset as dt
import torch
import torch.nn as nn
import config as config
from Transformer import Multi_attention_heads as Matt
from Transformer import Transformer_block as TRA
from Transformer import Transformer as Transformer
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Apply the initialization to your model
def train(device,confi,dir,load_state,load_model):    #warnings.filterwarnings("ignore")
    #config = get_config()
   # train_model(config)    
    #introduce_device
    #wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

    data = pd.read_csv('EURUSD60.csv')

    print(data.head())
    data=data['Close']#[1200:]
    data_min=np.min(data)
    data_max=np.max(data)
    data_norm=(data-data_min)/(data_max-data_min)
    data_norm = torch.tensor(data_norm.values)
    print('data_norm.shape',data_norm.shape)
    n = int(confi['split_train']*len(data_norm)) # first 90% will be train, rest val
    n_val=int(confi['split_val']*len(data_norm))
    train_data = data_norm[:n].float() ##shape(B,T) ,, (batch,sequence length)
    val_data = data_norm[n:n_val].float()
    test_data=data_norm[n_val:].float()

    print('train_data.shape:' + str(train_data.shape))
    print('val_data.shape:' + str(val_data.shape))
    print('test_data.shape:' + str(test_data.shape))

    # Create a custom dataset
    tr_dataset = dt.get_ds(train_data)
    val_dataset= dt.get_ds(val_data)
    test_dataset= dt.get_ds(test_data)

    tr_dataloader = DataLoader(tr_dataset, batch_size=confi['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=confi['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=confi['batch_size'], shuffle=True)
   
    print('tr_dataloader.shape',len(tr_dataloader))
    print('val_datloade.shape',len(val_dataloader))
    print('test_dataloader.shape',len(test_dataloader))

    model=Transformer.DecoderBlock(device)
    model=model.to(device)

    if load_model:
        print('Loading_model')
        state_dict_path=load_state
        state_dict = torch.load(state_dict_path,map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    else:
        print('New_training___')             
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    #print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=confi['lr'])
    initial_epoch=0

    for epoch in range(initial_epoch, confi['num_epochs']):
        torch.cuda.empty_cache()
        model.train() 
        iter=0        
        for batch in tr_dataloader:
            train_input=batch['input_seq'].to(device)
            train_tgt=batch['target_seq'].to(device)        
            pred, loss = model(train_input, train_tgt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()  
            # make a folder for saving the model 
            if not os.path.exists(dir):
                os.makedirs(dir)      
            if iter % 500 == 0:
                torch.save(model , dir+'model_iter_'+str(iter)+'.pth')
                torch.save(model.state_dict(), dir+'mode_state_iter_'+str(iter)+'.pt') 
                torch.save(optimizer.state_dict(), dir+'optimizer_state_dict'+str(iter)+'.pt')           

            print(f"epoch {epoch}: step {iter}: of_total_iter {len(tr_dataloader)}: train loss {loss:.4f}:", end="\r")


            # every once in a while evaluate the loss on train and val sets
                #loss_val=model(val_input, val_tgt)[1]
            #print(f"step {iter}: train loss {estimate_loss.loss(model,confi['eval_iter'],train_input, train_tgt):.4f},val loss {estimate_loss.loss(model,confi['eval_iter'],val_input, val_tgt):.4f}")
            ## we evaluate at each epoch for the half of the number of iteration in the trainging
            if iter % confi['eval_iter'] == 0 or iter == 2:
                eval_loss=evaluate(model,val_dataloader,device,len(val_dataloader))
                print('___eval_loss',eval_loss)
            del batch              
               # eval=evaluate(model,val_dataloader,device)
            #if iter % confi['eval_iter'] == 0 or iter == 3:
            iter=iter+1             

def evaluate(model,val_dataloader,device,total_iter):
    model.eval()  # Set the model to evaluation mode
    val_loss_list=[]
    iterr=0
    print('Validation____________')
    with torch.no_grad():  # Disable gradient computation
        for batch in val_dataloader:
            val_input=batch['input_seq'].to(device)
            val_tgt=batch['target_seq'].to(device)
            # Perform forward pass without gradient computation
            pred, val_loss = model(val_input, val_tgt)
            val_loss_list.append(val_loss)
            iterr=iterr+1
            print(f" val loss {val_loss:.4f}: iter {iterr}: of_total_iter {total_iter}", end="\r")
            del batch
        av_loss=sum(val_loss_list) / len(val_loss_list)
        print('av_loss',av_loss)

    return av_loss
