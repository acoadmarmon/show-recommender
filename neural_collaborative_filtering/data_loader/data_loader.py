import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import numpy as np

class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = user_id, item_id, rating
        """
        assert 'user_id' in ratings.columns
        assert 'item_id' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['user_id'].unique())
        self.item_pool = set(self.ratings['item_id'].unique())

        # create negative item samples for NCF learning
        self.preprocess_ratings = self._sample_negative(self.preprocess_ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        #ratings = deepcopy(ratings)
        #ratings['rating'][(ratings['rating'] > 0.0) & (ratings['rating'] < 6.0)] = -2.0
        ratings['rating'][ratings['rating'] >= -1.0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split """

        ratings['rank'] = np.random.randint(1, 100, ratings.shape[0])
        ratings['rank_latest'] = ratings.groupby(['user_id'])['rank'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['user_id'].nunique() == test['user_id'].nunique()
        return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = pd.DataFrame(ratings.groupby('user_id')['item_id'].apply(set)).rename(
            columns={'item_id': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 20))
        interacted_with = interact_status.explode('negative_samples')
        interacted_with['rating'] = 0.0
        interacted_with = interacted_with.drop(['interacted_items', 'negative_items'], axis=1)
        interacted_with = interacted_with.rename(columns={'negative_samples': 'item_id'}).reset_index()
        return pd.concat([interacted_with, ratings])

    def instance_a_train_loader(self, batch_size):
        """instance train loader for one training epoch"""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(list(self.train_ratings['user_id'])),
                                        item_tensor=torch.LongTensor(list(self.train_ratings['item_id'])),
                                        target_tensor=torch.FloatTensor(list(self.train_ratings['rating'])))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """create evaluate data"""
        interact_status = pd.DataFrame(self.test_ratings.groupby('user_id')['item_id'].apply(set)).rename(
            columns={'item_id': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 20))
        interacted_with = interact_status.explode('negative_samples')
        interacted_with['rating'] = 0.0
        interacted_with = interacted_with.drop(['interacted_items', 'negative_items'], axis=1)
        interacted_with = interacted_with.rename(columns={'negative_samples': 'item_id'}).reset_index()
        return [torch.LongTensor(list(self.test_ratings['user_id'])), torch.LongTensor(list(self.test_ratings['item_id'])), torch.LongTensor(list(interacted_with['user_id'])),
                torch.LongTensor(list(interacted_with['item_id']))]
