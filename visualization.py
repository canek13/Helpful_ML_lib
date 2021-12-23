import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt



def _get_str_features(data):
    """
    Returns 
    -------
    cat_features: list of column names {, cat_features_indexes:}
    """
    list_f = [col for col in data.columns if data[col].dtype == object or 
              type(data.loc[data[col].notna(), col].iloc[0]) == str]
            
    return list_f
    
def show_values_on_bars(axs, h_v="v", space=0.4):
    '''
    Shows values on bars.

    Parameters
    ----------
    axs: axis of plot
    h_v: {'h', 'v'}
        Whether the barplot is horizontal or vertical. 
        "h" represents the Horizontal barplot, 
        "v" represents the Vertical barplot.
    space: float,
        The space between value text and the top edge of the bar. 
        Only works for horizontal mode.

    Returns
    -------
    None

    See also
    --------
    https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
    https://datavizpyr.com/how-to-annotate-bars-in-barplot-with-matplotlib-in-python/

    Examples
    --------
    >>> fig, axis = plt.subplots(1,1)
    >>> sns.barplot(x=data, y=target, orient='h', ax=axis)
    >>> show_values_on_bars(axis, h_v='h', space=0.5)
    '''
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = np.round(float(p.get_height()), 2)
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = np.round(float(p.get_width()), 2)
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def plot_bar(data, 
            column_name_float: str, column_name_str: str, 
            top_limit_display: int=30, 
            sort=True, 
            stratify=None, 
            figsize=(5, 10), 
            orient='h'
            ):
    df = data.copy()
    
    if stratify is not None:
        if type(stratify) != str:
            data['stratify'] = stratify
            sort_by_col = 'stratify'
        else:
            sort_by_col = stratify
        
        df = df.sort_values(by=sort_by_col, 
                                ascending=False, 
                                na_position='last',
                               )
    #TODO sort
    if sort:
        df.sort_values(by=column_name_float,
                       ascending=False,
                       na_position='last',
                       inplace=True
                      )
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    sns.barplot(x=df[column_name_float][:top_limit_display],
                y=df[column_name_str][:top_limit_display],
                orient=orient,
                ax=ax,
               )
    show_values_on_bars(ax, h_v=orient)
    
    plt.yticks(fontsize=max(figsize) + 2)
    plt.xticks(fontsize=max(figsize) + 2)
    plt.xlabel(column_name_float, fontsize=max(figsize) + 5)
    plt.ylabel(column_name_str, fontsize=max(figsize) + 5)
    plt.show()

def plot_histograms(data, columns_draw=None, bins='auto'):
    if columns_draw is not None:
        cols = columns_draw
    else:
        cols = data.columns
    c = len(cols)
    
    fig, axis = plt.subplots(nrows=c, figsize=(8, 5 * c))
    for i, col in enumerate(cols):
        sns.histplot(data[col], ax=axis[i], bins=bins)