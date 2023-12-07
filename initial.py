import torch
import torch.nn as nn
import train as tr
from config import get_config
import time
import pandas as pd

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    dir='save_models'+'/'
    config=get_config()
    start_time = time.time()
    load_state='64_batch_250_seqmode_state_iter_31000.pt'
    tr.train(device,config,dir,load_state,load_model=False)
    total_time = time.time() - start_time 
    print('total_time=',total_time)
    time_list=[total_time]
    df = pd.DataFrame(time_list)
    df.to_csv(dir+'time'+'.csv', header=False, index=False)  




