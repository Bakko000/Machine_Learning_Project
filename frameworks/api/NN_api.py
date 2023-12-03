import pandas as pd


# Name of the columns of the Data sets
columns_name = ['target', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'id']




def load_data(path: str):
    '''
        Returns the DataFrame associated to the Data set found at path \"path\".\n
        - path: path to the CSV file with data.
    '''
    return pd.read_csv(filepath_or_buffer=path, names=columns_name, delimiter=' ')




def split_data(data: pd.DataFrame, target_col='target', drop_cols=['target','id']):
    '''
        Returns a tuple of two new DataFrames: (x,y).\n
        - x: is like \"df\" without the columns specified in the list \"drop_cols\".\n
        - y: is the column indentified by the key \"target_col\".\n
        The parameters are:\n
        - df: the input DataFrame.\n
        - target_col: name of the target column (default='target').\n
        - drop_cols: list of columns name to drop (default=['target','id']).
    '''
    y = data[target_col].copy(deep=True)
    x = data.drop(columns=drop_cols, axis=1).copy(deep=True)
    return x, y




def one_hot_encoding(data: pd.DataFrame):
    '''
        Returns the DataFrame got by appling the 1-Hot Encoding to the DataFrame
        passed as parameter.\n
        - df: the DataFrame to whom is applied the 1-Hot Encoding.
    '''

    # Creation of a Deep Copy of the original DataFrame
    df = data.copy(deep=True)

    # For each column we apply the dummies method
    for column in data.columns:

        # Applies one-hot encoding to current column and renames them
        one_hot_cols = pd.get_dummies(df[column], dtype=float)
        one_hot_cols = one_hot_cols.set_axis([column+'_'+str(col) for col in one_hot_cols.columns], axis=1)

        # Drops of the original column and adds the new columns
        df = df.drop(column, axis=1)
        df = pd.concat([df,one_hot_cols], axis=1)

    return df
