import torch
from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence


class RatingDataset(Dataset):
    def __init__(self, data, max_genres=10):
        self.data = data
        self.max_genres = max_genres

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        data_item = self.data.iloc[index]
        
        user_embed_id = data_item["user_embed_id"]
        movie_embed_id = data_item["movie_embed_id"]
        # genres_embed_ids = data_item["genres_embed_ids"]
        # genres_embed_ids = [torch.tensor(ids) for ids in genres_embed_ids]
        # padded_genres_embed_ids = pad_sequence(
        #     genres_embed_ids, batch_first=True, padding_value=0)
        
        # padded_genres_embed_ids = padded_genres_embed_ids[:, :self.max_genres]

        rating = self.data.iloc[index]["rating"]

        sample = {
            "user_embed_id": torch.tensor(user_embed_id, dtype=torch.long),
            "movie_embed_id": torch.tensor(movie_embed_id, dtype=torch.long),
            # "genres_embed_ids": padded_genres_embed_ids,
            "rating": torch.tensor(rating, dtype=torch.float),
        }

        return sample
