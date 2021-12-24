import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def _get_str_features(data):
    """
    Returns 
    -------
    cat_features: list of column names {, cat_features_indexes:}
    """
    list_f = [col for col in data.columns if data[col].dtype == object or 
              type(data.loc[data[col].notna(), col].iloc[0]) == str]
            
    return list_f

def describe_values_dataframe(df):
    '''
    Describes data in DataFrame: 
    unique values, not NaN values count,
    NaN values count, NaN valeus percent

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame to describe
    
    Returns
    -------
    Described features in data: pd.DataFrame

    See also
    --------
    For better representation in Jupyter Notebook:
    pd.set_option('display.max_columns', None)
    pd.set_option("display.max_rows", 200)
    # pd.reset_option("display.max_rows")
    '''
    nan_percent = (pd.isna(df).sum() / df.shape[0]) * 100
    nan_count = pd.isna(df).sum().astype(int)
    no_nan_count = pd.notna(df).sum().astype(int)
    unique_values_cnt = [df[col].unique().size for col in df.columns]
    shape = [df[col].shape[0] for col in df.columns]
    
    cols = np.round([shape, unique_values_cnt, no_nan_count, nan_count, nan_percent], 1)
    index_names = ['shape', 'uniq_values', 'no_nan', 'nan', 'nan%']
    col_names = df.columns
    d = pd.DataFrame(data=cols,
                    index=index_names,
                    columns=col_names)
    return d.transpose()

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Iterate through all the columns of a dataframe 
    and modify the data type to reduce memory usage.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe to reduce memory usage
    
    Returns
    -------
    df: pandas.DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if not is_categorical_column(df[col]): # col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def is_categorical_column(seria, limit_uniq_cnt: int=8) -> bool:
    '''
    Checks if seria is categorical column.
    Also prints if ammount of unique values is small.

    Parameters
    ----------
    seria: pd.Series
        Pandas seria to check.
    limit_uniq_cnt: int, optional
        Checks for unique values in seria.
        If uniq_values less than parameter prints log.
        if None: NO printing.
    
    Returns
    -------
    bool
    '''
    if limit_uniq_cnt is not None \
        and seria.unique().size <= limit_uniq_cnt:
        print(seria.name + ' has few unique values')
    
    if seria.notna().any() and (seria.dtype == object or \
        isinstance(seria[seria.notna()].iloc[0],  str) ):
        return True
    else:
        return False

def get_cat_features(df, limit_uniq_cnt: int=None, return_only_cat: bool=True):
    '''
    Finds columns containing str objects 
    or column.dtype is object
    
    Uses 'is_categorical_column' func

    Parameters
    ----------
    df: pandas.DataFrame
    limit_uniq_cnt: int, optional
        Checks for unique values in seria.
        If uniq_values less than parameter prints log.
        if None: NO printing.
    return_only_cat: bool,
        Whether to return only categorical features names or
        cat_features, cat_features_indexes,
        non_cat_features, non_cat_features_indexes

    Returns
    -------
    categorical_features_column_name, categorical_features_idx_column,
    not_categorical_features_column_name, not_categorical_features_idx_column: lists
        idx -- index of column
    '''
    cat_features = []
    cat_features_i = []

    for i, name in enumerate(df.columns):
        if is_categorical_column(df[name], limit_uniq_cnt=limit_uniq_cnt):
            cat_features.append(name)
            cat_features_i.append(i)

    not_cat_features = [name for name in df.columns if name not in cat_features]
    not_cat_features_i = [i for i, name in enumerate(df.columns) if name not in cat_features]
    
    if return_only_cat:
        return cat_features

    return cat_features, cat_features_i,\
            not_cat_features, not_cat_features_i

def multiply_lists(l1, l2):
    '''
    Concatenate STRING elemnts from 2 lists.
    Every elemnt from l1 folds with every element from l2

    Parameters
    ----------
    l1: list
    l2: list

    Returns
    -------
    list

    Warning
    -------
    May work with numeric data but be careful with result

    Examples
    --------
    >>> multiply_lists(['a', 'b', 'c'], ['d', 'e'])
    ['ad', 'ae', 'bd', 'be', 'cd', 'ce']
    '''
    if isinstance(l1[0], str) or isinstance(l2[0], str):
        return [str(first) + str(second) 
                for first in l1 
                for second in l2]
    else:
        return [first + second 
                for first in l1 
                for second in l2]

def print_column_nans_in_dataframe(dataframe):
    '''
    Print how many nans are in every dataframe's column

    Parameters
    ----------
    dataframe: pandas.DataFrame
    '''
    for col, nan in zip(dataframe.columns, pd.isna(dataframe).sum().tolist()):
        print(col, nan)
    return
        
def print_column_infinity_in_dataframe(dataframe):
    '''
    Print how many infinity values are in every dataframe's column

    Parameters
    ----------
    dataframe: pandas.DataFrame
    '''
    try:
        for col, inf in zip(dataframe.columns, np.isinf(dataframe).sum().tolist()):
            print(col, inf)
    except TypeError:
        print('Non Numeric columns in DataFrame')
        print('Using "dataframe.select_dtypes(incude=np.number)" !!!', end='\n\n')
        
        for col, inf in zip(dataframe.select_dtypes(include=np.number).columns, 
                            np.isinf(dataframe.select_dtypes(include=np.number))
                            .sum().tolist()):
            print(col, inf)
    return

def get_permutation_indexes_train_test(df_shape: tuple, 
                                       test_size: float=0.2,
                                       random_state=None):
    '''
    Make train mask with test size=test_size for dataset with shape df_shape
    Only df_shape[0] is used

    Parameters
    ----------
    df_shape: tuple
        Shape of hole dataframe to split for train/test
    test_size: float, optional
        Default is 0.2
        Train size will be 0.8 of dataset
    
    Returns
    -------
    Mask: ndarray
        Where '1' stands as train markers.
        Sum of array == 0.8 * df_shape[0]
    '''
    rows, columns = df_shape

    np.random.seed(random_state)
    test_indexes = np.random.choice(rows, size=int(rows * test_size), replace=False)
    
    mask = np.ones(rows, dtype=np.int8)
    mask[test_indexes] = 0
    
    return mask == 1

def test_catboost(data, 
                  drop_columns=['globalcode', 'onekeyid', 'tetlong',], 
                  add_cat_features=[], 
                  target_column_name=None, 
                  train_mask=None,
                  test_size: float=0.25,
                  metric=None,
                  regressor_catboost: bool=True,
                  return_proba: bool=False,
                  **cat_kwargs,
                  ):
    '''
    Test the data with CatBoostRegressor.
    Fits model on data.loc[train_mask]
    And print metrics (R2_score and RMSE) on train and test set

    Parameters
    ----------
    data: pandas.DataFrame
        Input data
    drop_columns: list or string, optional
        Define what columns to drop from data.
        Default is ['globalcode', 'onekeyid', 'tetlong'] as non-informative
    add_cat_features: list, optional
        Whether to add categorical columns to base return of get_cat_features func.
        For exmple if some column is label encoded, the func get_cat_features won't
        detect this column as categorical.
    target_column_name: string, required
        The name of target column in data.
        It will be dropped before train/test
    train_mask: bool or int ndarray
        Where '1' or 'True' will be train set for Catboost
    test_size: float,
        if train mask is None 
        this parameter will be set for train_test_split.test_size
    metric: callable,
        If provided, must be callable and respond sklearns' metrics:
        metric(y_true, y_pred)
    regressor_catboost: bool, deafault=True
        Whether to use CatboostRegressor or CatboostClassifier.
        In deafault case CatboostRegressor
    return_proba: bool, default=False
        Whether to return probability of prediction.
        Make sense only in classfication case.
        If True, regressor_catboost must be False
    **cat_kwargs: CatBoostRegressor additional kwargs

    Returns
    -------
    CatBoostRegressor: object, fitted on data
    FeatureNames: list of data's column names involved in train process
    '''
    df = data.copy()
    target = df[target_column_name].values
    
    df = df.drop(columns=drop_columns + [target_column_name])
    cat_f, cat_f_i, \
    not_cat_f, not_cat_f_i = get_cat_features(df, 
                                            return_only_cat=False)
    
    df[cat_f] = df[cat_f].fillna('NA')
    
    cat_f += add_cat_features
    if len(cat_f) != len(cat_f_i):
        cat_f_i += [i for i, name in enumerate(df.columns) if name in add_cat_features]
    print('categotical features = ', cat_f) #, cat_f_i)
    
    #df[cat_f] = df[cat_f].fillna('NA')
    
    if train_mask is None:
        train_mask = get_permutation_indexes_train_test(df.shape, 
                                                        test_size=test_size, 
                                                        random_state=42)

    X_train = df.values[train_mask]
    y_train = target[train_mask]
    X_test = df.values[~train_mask]
    y_test = target[~train_mask]

    if regressor_catboost:
        c = CatBoostRegressor(**cat_kwargs, cat_features=cat_f_i, logging_level='Silent')
    else:
        c = CatBoostClassifier(**cat_kwargs, cat_features=cat_f_i, logging_level='Silent')
    
    c.fit(X_train, y_train)
    
    if return_proba:
        if regressor_catboost:
            raise ValueError('Regressor can predict probability. Change "return_proba" or "regressor_catboost"')
        y_pred = c.predict_proba(X_test)[:, 1]
        y_pred_train = c.predict_proba(X_train)[:, 1]
    else:
        y_pred = c.predict(X_test)
        y_pred_train = c.predict(X_train)
    
    if metric is not None:
        print('Train: ', metric(y_train, y_pred_train))
        print('Test: ', metric(y_test, y_pred))
    else:
        print('RMSE test = ', mean_squared_error(y_test, y_pred, squared=False),\
            'RMSE train = ', mean_squared_error(y_train, y_pred_train, squared=False))
        print('R2 test = ', r2_score(y_test, y_pred), 'R2 train = ', r2_score(y_train, y_pred_train))
    
    return c, df.columns.tolist()

def fillna_inplace_with_median(df, columns_to_groupby, columns_to_fillna):
    '''
    Fills nans in dataframe with median inplace!

    Parameters
    ----------
    df: pandas.DataFrame
        Input data
    columns_to_groupby: list or ndarray of str
        Column names of input data to group on
    columns_to_fillna: list or ndarray of str
        Column names of input data to fill Nan values

    Returns
    -------
    None

    See Also
    --------
    fillinf_inplace_with_median: fills infnities with median inplace
    '''
    for column_to_fillna in columns_to_fillna:
        #for column_to_groupby in columns_to_groupby:
            df[column_to_fillna] = df[column_to_fillna].fillna(
                                    df.groupby(columns_to_groupby)[column_to_fillna].
                                    transform('median'))

def fillinf_inplace_with_median(df, columns_to_groupby, columns_to_fillinf):
    '''
    Fills infinities in dataframe with median inplace!

    Parameters
    ----------
    df: pandas.DataFrame
        Input data
    columns_to_groupby: list or ndarray of str
        Column names of input data to group on
    columns_to_fillna: list or ndarray of str
        Column names of input data to fill inf values

    Returns
    -------
    None

    See Also
    --------
    fillna_inplace_with_median: fills NaNs with median inplace
    '''
    for column_to_fillinf in columns_to_fillinf:
        
        df[column_to_fillinf][np.isinf(df[column_to_fillinf])] = df.groupby(
                                                                        columns_to_groupby
                                                                        )[column_to_fillinf].transform('median')
    return

def test_estimator(estimator, 
                   data_X,
                   data_y,
                   drop_columns=[], 
                   add_cat_features=[], 
                   train_mask=None,
                   fillna_value: float=0,
                   test_size: float=0.25,
                   metric=None,
                   return_proba: bool=False,
                   #**estimator_kwargs
                  ):
    '''
    Test the data with custom estimator.
    Fits model on data.loc[train_mask]
    And print metrics (R2_score and RMSE) on train and test set

    Parameters
    ----------
    estimator: class instance
        Already created estimator
    data_X: pandas.DataFrame
        Input data without target column
    data_y: pandas.Series
        Target column
    drop_columns: list or string, optional
        Define what columns to drop from data.
        Default is ['globalcode', 'onekeyid', 'tetlong'] as non-informative
    add_cat_features: list, optional
        Whether to add categorical columns to base return of get_cat_features func.
        For exmple if some column is label encoded, the func get_cat_features won't
        detect this column as categorical.
    train_mask: bool or int ndarray of data_y.size
        Where '1' or 'True' will be train set for estimator.
        Must be the size of data_y!!!
    fillna_value: float,
        Value to fill NANs in dataframe data_X
    test_size: float,
        if train mask is None 
        this parameter will be set for train_test_split.test_size
    metric: callable,
        If provided, must be callable and respond sklearns' metrics:
        metric(y_true, y_pred)
    return_proba: bool, default=False
        Whether to return probability of prediction.
        Make sense only in classfication case.

    Returns
    -------
    estimator: instance, fitted on data
    FeatureNames: list of data's column names involved in train process

    '''
    df = data_X.copy()
    target = data_y.values
    
    df = df.drop(columns=drop_columns)
    cat_f, cat_f_i, \
    not_cat_f, not_cat_f_i = get_cat_features(df,
                                            return_only_cat=False)
    
    cat_f += add_cat_features
    if len(cat_f) != len(cat_f_i):
        cat_f_i += [i for i, name in enumerate(df.columns) if name in add_cat_features]
    print('categotical features = ', cat_f)
    
    #df[cat_f] = df[cat_f].fillna('NA')
    df = df.fillna(fillna_value)
    
    if train_mask is None:
        train_mask = get_permutation_indexes_train_test(df.shape, 
                                                        test_size=test_size,
                                                        random_state=42)

    X_train = df.values[train_mask]
    y_train = target[train_mask]
    X_test = df.values[~train_mask]
    y_test = target[~train_mask]

    r = estimator # (**estimator_kwargs)
    r.fit(X_train, y_train)
    y_pred = r.predict(X_test)
    
    #y_pred_train = r.predict(X_train)
    
    if return_proba:
        y_pred = r.predict_proba(X_test)[:, 1]
        y_pred_train = r.predict_proba(X_train)[:, 1]
    else:
        y_pred = r.predict(X_test)
        y_pred_train = r.predict(X_train)

    if metric is not None:
        print('Train: ', metric(y_train, y_pred_train))
        print('Test: ', metric(y_test, y_pred))
    else:
        print('test = ', mean_squared_error(y_test, y_pred, squared=False),\
            ' train = ', mean_squared_error(y_train, y_pred_train, squared=False))
        print('test = ', r2_score(y_test, y_pred), ' train = ', r2_score(y_train, y_pred_train))
    
    return r, df.columns.tolist()
