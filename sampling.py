import numpy as np
import pandas as pd

from tqdm import tqdm

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def split_data_independently(data, 
                                target_col_name: str, 
                                stratify_col=None, 
                                test_size=0.3, 
                                drop_cols=None,
                                random_state=42,
                               ):
    '''
    Returns data splitted into train and test using sklearn's train_test_split
    
    Returns
    -------
    X_train, X_test, y_train, y_test: pd.DataFrame
    '''
    dr_cols = [target_col_name] + drop_cols if drop_cols is not None else [target_col_name]
    return train_test_split(data.drop(columns=dr_cols),
                           data[target_col_name],
                           stratify=stratify_col,
                           test_size=test_size,
                           random_state=random_state)


def split_different(data,
                     target_col_name: str, 
                     col_split_name: str, 
                     test_size=0.3, 
                     drop_cols=None,
                    ):
    '''
    Returns data splitted into train and test.
    Sampling different col_split_name unique values.
    Intersection of train[col_split_name] and test[col_split_name] = 0
    
    Returns
    -------
    X_train, X_test, y_train, y_test: pd.DataFrame
    '''
    col_split_uniq = data[col_split_name].unique()
    train_id = random.sample(col_split_uniq.tolist(), int(col_split_uniq.shape[0] * (1 - test_size)))
    test_id = np.setdiff1d(col_split_uniq, train_id, assume_unique=True)
    
    train = data[data[col_split_name].isin(train_id)]
    test = data[data[col_split_name].isin(test_id)]
    
    dr_cols = [target_col_name] + drop_cols if drop_cols is not None else [target_col_name]
    return train.drop(columns=dr_cols), test.drop(columns=dr_cols), \
            train[target_col_name], test[target_col_name]


class FullyFledgedSelection:
    def __init__(self,
                 data: pd.DataFrame,
                 target_name: str,
                 estimator=LGBMClassifier,
                 clf_metric=roc_auc_score,
                 test_size: float=0.3,
                 n_iter=10,
                 **estimator_kwargs,
                ):
        self.data = data
        self.target_name = target_name
        
        self.cnt_samples, _ = self.data.shape
        
        self.test_size = test_size
        self.n_iter = n_iter
        
        self.best_test_idx = -1
        self.samples_marking_permutation = self._generate_permutations_marking_test()
        print('Permutations generated')
        
        self.estimator = estimator(**estimator_kwargs)
        self.metric = clf_metric
        self.lowest_score = 0.5
        
    def make_selection(self):
        """
        Returns
        -------
        Train, test
        """
        for i in tqdm(range(self.n_iter)):
            y_test, y_pred = self._fit_predict(idx_permut=i)
            
            if abs(0.5 - self.metric(y_test, y_pred)) < self.lowest_score:
                
                self.lowest_score = abs(0.5 - self.metric(y_test, y_pred))
                self.best_test_idx = i
        
        print('lowest score = ', self.lowest_score)
        
        samples_test_i = self.samples_marking_permutation[self.best_test_idx]
        samples_train_i = np.setdiff1d(np.arange(self.cnt_samples),
                                       samples_test_i,
                                       assume_unique=True,
                                      )
        test = self.data.iloc[samples_test_i]
        train = self.data.iloc[samples_train_i]
        
        return train, test
    
    def _fit_predict(self, idx_permut):
        samples_zero_i = self.samples_marking_permutation[idx_permut]
        samples_ones_i = np.setdiff1d(np.arange(self.cnt_samples),
                                      samples_zero_i,
                                      assume_unique=True,
                                     )
        samples_zero = self.data.iloc[samples_zero_i]
        samples_ones = self.data.iloc[samples_ones_i]
        
        samples_zero.loc[:, 'target'] = 0
        samples_ones.loc[:, 'target'] = 1
        
        X_train, X_test, \
        y_train, y_test = self._create_train_test(samples_ones, samples_zero)
        
        self.estimator.fit(X_train, y_train)
        y_pred = self.estimator.predict(X_test)
        
        return y_test, y_pred
    
    def _create_train_test(self, train, test):
        data = train.append(test)
        
        return train_test_split(data.drop(columns=['target', self.target_name]).values,
                                data['target'].values,
                                test_size=self.test_size,
                                random_state=None,
                                stratify=data['target'].values,
                               )
    def _generate_permutations_marking_test(self):
        cols_matr = int(self.cnt_samples * self.test_size)
        matr_permutation = np.zeros((self.n_iter, cols_matr), dtype=int)
        
        for i in range(self.n_iter):
            matr_permutation[i] = np.random.choice(self.cnt_samples,
                                                   size=cols_matr,
                                                   replace=False,
                                                  )
        return matr_permutation