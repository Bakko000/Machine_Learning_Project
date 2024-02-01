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
            Choose a random value for each list associated with its own key and returns a new dictionary \
            with that association.\n
            - params: dictionary with parameters (keys = parameter_name, values = possible_values_list).
        '''
        new_params = {}
        for key in params.keys():
            new_params[key] = np.random.choice(params[key])
        return new_params


    def set_params_combinations(self, params: dict) -> dict:
        '''
            Creates and saves into the class instance a list with all the possible combinations of parameters \
            in the dictionary \"params\".\n
            - params: dictionary with parameters (keys = parameter_name, values = possible_values_list).
        '''
        self.current_params_index = 0
        self.params_index_dict = {}
        self.params_combinations = []
        for key in params.keys():
            self.params_index_dict[key] = 0 # current_index for that key
        while sum([index+1 for _, index in self.params_index_dict.items()]) != sum(len(val_list) for _, val_list in params.items()):
            params_i = {}
            for key, i in self.params_index_dict.items():
                params_i[key] = params[key][i]
            self.params_combinations.append(params_i)
            for key in self.params_index_dict.keys():
                self.params_index_dict[key] += 1
                if self.params_index_dict[key] < len(params[key]):
                    break
                self.params_index_dict[key] = 0
        params_i = {}
        for key, i in self.params_index_dict.items():
            params_i[key] = params[key][i]
        self.params_combinations.append(params_i)


    def get_params_combinations(self) -> dict:
        '''
            Returns the list of all the combinations (as doctionaries) got in the last call of the method of \
            set_params_combinations.
        '''
        return self.params_combinations


    def load_data(self, path: str, delimiter=' ') -> pd.DataFrame:
        '''
            Returns the DataFrame associated to the Data set found at path \"path\".\n
            - path: path to the CSV file with data.\n
            - delimiter: character used as delimiter in the CSV file.
        '''
        return pd.read_csv(filepath_or_buffer=path, names=self.columns_name, delimiter=delimiter, comment='#')
    

    def write_data(self, filename: str, id_list: list, data: list[list], cols_name: list):
        '''
            Creates a file CSV with the dataset passed as parameter and some static comments.\n
            - filename: name of the CSV file.\n
            - id_list: list of IDs which will be added as the first column of the CSV file.\n
            - data: list of data (matrix of 2 dimensions) which will be added to the CSV file.\n
            - cols_name: list of names of the columns.
        '''
        nickname = "EmmElle"

        # Opens the CSV file
        with open(filename, 'w') as f:

            # Writes some comments
            f.write('# Creators: Emad Chelhi - Gianluca Panzani - Corrado Baccheschi\n')
            f.write(f'# Nickname: {nickname}\n')
            f.write(f'# Dataset\'s name: {nickname}_3-th_place_dataset\n')
            f.write(f'# Date: 31 Jan 2024\n')

            # Writes columns name
            for i, col in enumerate(cols_name):
                if i < len(cols_name)-1:
                    f.write(f'{col},')
                else:
                    f.write(f'{col}\n')
            
            # Writes IDs and data
            for id, row in zip(id_list,data):
                f.write(f'{id},')
                for i, elem in enumerate(row):
                    if i < len(row)-1:
                        f.write(f'{elem},')
                    else:
                        f.write(f'{elem}\n')


    def split_data(self, data: pd.DataFrame, cols_name_split: list, rows_split_perc=1):
        '''
            It makes the split of the columns passed in \"cols_name_split\" and the split of the rows based on the \
            percentage \"rows_split_perc\".\n
            If \"cols_name_split\" = [] -> the method returns: (data, None).\n
            else -> (data_splitted_x, data_splitted_y) with rows_split_perc=1 \
                or (data_splitted_x_train, data_splitted_y_train, data_splitted_x_val, data_splitted_y_val) with rows_split_perc!=1.\n
            So this method makes split on columns or on rows (or both).\n\n
            Returns a tuple of two new DataFrames: (x,y).\n
            - x: is like \"df\" without the columns specified in the list \"cols_name_split\".\n
            - y: are the columns indentified by the list \"cols_name_split\".\n
            or a tuple of this format (x_train, y_train, x_val, y_val) with:
            - x_train: is like \"df\" without the columns specified in the list \"cols_name_split\".\n
            - y_train: are the columns indentified by the list \"cols_name_split\" used for Training.\n
            - x_val: is like \"df\" without the columns specified in the list \"cols_name_split\".\n
            - y_val: are the columns indentified by the list \"cols_name_split\" used for Validation.\n\n
            The parameters are:\n
            - data: the input DataFrame.\n
            - cols_name_split: list of names of target columns.\n
            - rows_split_perc: percentage of the data to split for Training, and so the 1-rows_split_perc percentage for Validation.
        '''
        # Case of no columns split
        if cols_name_split == []:
            # Case of rows split
            if rows_split_perc != 1:
                return np.split(data, [int(data.shape[0] * rows_split_perc)], axis=0)
            # Case of no rows and no columns split
            else:
                raise ValueError
        
        # Columns split
        y = data[cols_name_split].copy(deep=True)
        x = data.drop(columns=cols_name_split,axis=1).copy(deep=True)

        # Case of only columns split
        if rows_split_perc == 1:
            return x, y
        
        # Case of both splits (rows split and columns split)
        else:
            x_train, x_val = np.split(x, [int(x.shape[0] * rows_split_perc)], axis=0)
            y_train, y_val = np.split(y, [int(y.shape[0] * rows_split_perc)], axis=0)
            return x_train, y_train, x_val, y_val


    def one_hot_encoding(self, data: pd.DataFrame):
        '''
            Returns the DataFrame got by appling the 1-Hot Encoding to the DataFrame \
            passed as parameter.\n
            - data: the DataFrame to whom is applied the 1-Hot Encoding.
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

