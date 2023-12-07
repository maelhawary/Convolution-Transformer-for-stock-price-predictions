import torch
from torch.utils.data import Dataset
import config as config

class get_ds(Dataset):
    def __init__(self, dt):
        self.confi=config.get_config()
        self.ds = dt
        self.sequence_length = self.confi['seq_len']

    def __len__(self):
        return len(self.ds)- self.sequence_length -1

    def __getitem__(self, index):
        input_seq = self.ds[index:index + self.sequence_length]
        target_seq = self.ds[index + self.sequence_length:index + self.sequence_length+1]
        return {
            "input_seq": input_seq,  
            "target_seq": target_seq,  
        }

