# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:57:01 2024

@author: 18370
"""

import numpy as np
import pandas as pd

df = pd.read_csv('testing_set.csv')


#%% Crossvalidation code
import numpy as np
import pandas as pd


# return three set of datas, not datas' index
class UserBasedGroupKFoldCV:
    def __init__(self, n_splits=5, random_state=None):
        # Initialize with the number of splits for each user and a random seed for reproducibility
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, df, group_column):
        # Set the random seed
        np.random.seed(self.random_state)
        
        # Shuffle the DataFrame globally to randomize the order of all rows
        df_shuffled = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Generate a group to indices map
        group_indices = df_shuffled.groupby(group_column).indices

        # Prepare the splits for each group
        results = []
        group_splits = {}

        for group, indices in group_indices.items():
            num_records = len(indices)
            if num_records >= self.n_splits:
                np.random.shuffle(indices)
                group_splits[group] = np.array_split(indices, self.n_splits)
            else:
                raise ValueError(f"Not enough records for user {group} to split into {self.n_splits} parts.")

        # Generate cross-validation folds
        for i in range(self.n_splits):
            train_indices = []
            test_indices = []
            val_indices = []

            for group, splits in group_splits.items():
                test_indices.extend(splits[i])
                val_indices.extend(splits[(i + 1) % self.n_splits])
                for j in range(self.n_splits):
                    if j != i and j != (i + 1) % self.n_splits:
                        train_indices.extend(splits[j])

            yield (df_shuffled.iloc[train_indices],
                   df_shuffled.iloc[test_indices],
                   df_shuffled.iloc[val_indices])

# Usage example
# df = pd.read_csv('your_data.csv')
# user_kf = UserBasedGroupKFoldCV(n_splits=5, random_state=42)
# for train_set, test_set, val_set in user_kf.split(df, 'user_id'):
#     print("Training set:", train_set.shape)
#     print("Testing set:", test_set.shape)
#     print("Validation set:", val_set.shape)
