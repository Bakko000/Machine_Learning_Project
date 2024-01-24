import pandas as pd
import numpy as np
from itertools import product



class DataHandler():
    '''
        Class which offers methods to handle DataFrames oriented to ML problems.
    '''

    def __init__(self, columns_name=[]):
        self.columns_name = columns_name
        self.params_combinations = []
    

    def random_dictionary(self, params: dict) -> dict:
        '''
        '''
        new_params = {}
        for key in params.keys():
            new_params[key] = np.random.choice(params[key])
        return new_params


    def set_params_combinations(self, params: dict) -> dict:
        '''
            
        '''
        self.current_params_index = 0
        self.params_index_dict = {}
        self.params_combinations = []
        for key in params.keys():
            self.params_index_dict[key] = 0 # current_index for that key
        while sum([index+1 for key, index in self.params_index_dict.items()]) != sum(len(val_list) for key, val_list in params.items()):
            params_i = {}
            for key, i in self.params_index_dict.items():
                params_i[key] = params[key][i]
            for key in self.params_index_dict.keys():
                self.params_index_dict[key] += 1
                if self.params_index_dict[key] < len(params[key]):
                    break
                self.params_index_dict[key] = 0
            self.params_combinations.append(params_i)
        
        for combination in self.params_combinations:
            print(combination)
    

    def get_params_combinations(self) -> dict:
        '''
        '''
        return self.params_combinations


    def load_data(self, path: str) -> pd.DataFrame:
        '''
            Returns the DataFrame associated to the Data set found at path \"path\".\n
            - path: path to the CSV file with data.
        '''
        return pd.read_csv(filepath_or_buffer=path, names=self.columns_name, delimiter=' ')


    def split_data(self, data: pd.DataFrame, target_col: str, drop_cols: list):
        '''
            Returns a tuple of two new DataFrames: (x,y).\n
            - x: is like \"df\" without the columns specified in the list \"drop_cols\".\n
            - y: is the column indentified by the key \"target_col\".\n
            The parameters are:\n
            - df: the input DataFrame.\n
            - target_col: name of the target column.\n
            - drop_cols: list of columns' name to drop.
        '''
        y = data[target_col].copy(deep=True)
        x = data.drop(columns=drop_cols, axis=1).copy(deep=True)
        return x, y


    def one_hot_encoding(self, data: pd.DataFrame):
        '''
            Returns the DataFrame got by appling the 1-Hot Encoding to the DataFrame \
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

