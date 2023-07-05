import torch
from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence


class RatingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        item_dict = self.data.iloc[index].to_dict()
        item_dict['target_rating'] = item_dict['rating_sequence'][-1]
        item_dict['rating_sequence'] = item_dict['rating_sequence'][:-1]
        item_dict['target_movie']= item_dict['movie_sequence'][-1]


        sample = {k:torch.tensor(v) for k,v in item_dict.items()}

        return sample
