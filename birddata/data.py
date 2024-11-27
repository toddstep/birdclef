import os

import pandas as pd
import constants
from preprocessing import duration

class Data:
    """Metadata of birdsong recordings.
    
    Arg:
        path (str):
            pathname to CSV file containing metadata
    """
    
    def __init__(self, path):
        self.df = pd.read_csv(path)
        if constants.is2020data:
            self.df['primary_label'] = self.df['ebird_code']
            self.df['filename'] = self.df['filename'].apply(lambda x: x.split('.')[0] + '.ogg')
            self.df['filename'] = self.df[['primary_label', 'filename']].apply(lambda x: '/'.join(x), axis=1)
        self.df['full_path'] = self.df['filename'].apply(lambda x: os.path.join(constants.base_dir, 'train_audio', x))
        self.competition_classes = sorted(self.df.primary_label.unique())
        self.num_classes = len(self.competition_classes)
        self.df['primary_index'] = self.df['primary_label'].apply(lambda x: self.competition_classes.index(x)).astype('int32')
        self.df['duration'] = self.df['full_path'].apply(duration)
        self.df, _ = Data.get_counts(self.df, 'primary_label_freq')
        print("NUM_CLASSES", self.num_classes)

    def get_folds(self, high_freq_thresh=25, valid_take_num=5):
        """Compute train/validation split of metadata
        
        Classes with a low number of samples are excluded from the validation set.
        While this may introduce a bias to the validation accuracy,
        it will provide more samples of such classes for training.
                
        Args:
            high_freq_thresh (int): The minimum number of samples for a class to be included in a validation split.
            valid_take_num (int): The number of samples to take for the validation split from each selected class.
        
        Returns:
            dict:
                'train': Dataframe with training elements
                'valid': Dataframe with validation elements
                'loss_weights': Weights (using the inverse of a class's frequency) for loss computation when training
        """
        if constants.is2020data:
            valid_rows = self.df['filename'].apply(lambda x: x.split('.')[-2][-3] == '0')
            valid_df = self.df[valid_rows]
        else:            
            high_freq = self.df['primary_label_freq'] >= high_freq_thresh
            valid_df = self.df[high_freq].groupby('primary_label').sample(n=valid_take_num, replace=False)
        train_df = self.df.drop(valid_df.index).sample(frac=1)
        if constants.is2020data:
            # keep_rows = self.df['filename'].apply(lambda x: 'XC313679.' not in x)
            keep_rows = self.df['filename'].apply(lambda x: x not in ['amegfi/XC313679.ogg', 'norwat/XC192964.ogg'])
            train_df = train_df[keep_rows]
        class_weight = self._get_class_weight(train_df)
        return {'train': train_df,
                'valid': valid_df,
                'class_weight': class_weight,}
    
    def _get_class_weight(self, df, max_weight=10.):
        _, counts = Data.get_counts(df, 'counts')
        class_weight = {idx: max_weight/counts[c] for idx, c in enumerate(self.competition_classes)}
        return class_weight

    @staticmethod
    def get_counts(df, count_col):
        """Counts the number of rows that contain each `primary_label`
        
        Args:
            df (DataFrame): dataframe that contains column `primary_label`
            count_col (str): column to contain count
        
        Returns:
            df (DataFrame): Table containing count column
            counts (Series): Counts for each `primary_label`
        
        """
        counts = df.groupby('primary_label').count()['filename'].rename(count_col)
        df = df.merge(counts, left_on='primary_label', right_on='primary_label').set_index(df.index)
        return df, counts


def get_train_filenames_per_index(df, classes):
    """Create lists of filenames associated with the training classes.
    
    Index `k` in the outer list contains the file in `df` for class with index `k`
    
    Arg:
        df (DataFrame): Table with filenames and primary_label columns
        classes (list): Classes ordered according to their index
    
    Returns:
        list: each element contains a list of filenames
    """
    #https://www.statology.org/pandas-groupby-list/
    filenames_per_label = df.groupby('primary_label')['full_path']
    filenames_per_label = dict(filenames_per_label.agg(list))
    filenames_list = [filenames_per_label[k] for k in classes]
    return filenames_list
