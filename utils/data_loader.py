'''
Date: 2024-11-30
Group: 5
Course: AAI-501: Intro to AI
'''

#Lets import the necessary modules 
import os
import pandas as pd
import kagglehub
from typing import Union

class DataLoader:
    '''
    Contains all methods necessary to perform
    the following:
    1.) Check and create a data folder.
    2.) Check and download the dataset.
    '''

    def __init__(self, data_folder_path:str, data_folder_name:str) -> None:
        '''
        Parameters:
            data_folder_path(str): desired storing data folder path
            data_folder_name(str): data folder name
        '''
        self.data_path = os.path.join(data_folder_path, data_folder_name)
        self.dataset_name = "credit_card_transactions.csv"

    def fetch_dataset(self) -> None:
        '''
        Instance method that would fetch the dataset 
        '''
        self._set_data_folder()
        if not self._check_existence(os.path.join(self.data_path, self.dataset_name)):
            print("download the dataset")
            path = kagglehub.dataset_download("priyamchoksi/credit-card-transactions-dataset")
            print(f"Path to dataset files:{path} \n \
                Please move the downloaded file to data folder, thank you :)")
        else:
            print("The dataset already exists")

    def get_dataset(self) -> Union[pd.DataFrame, None]:
        if not self._check_existence(os.path.join(self.data_path, self.dataset_name)):
            print(f"{self.dataset_name} is not in the folder, please move it here to retrieve it")
        else:
            return pd.read_csv(os.path.join(self.data_path, self.dataset_name))

    def _set_data_folder(self) -> None:
        '''
        Instance method that sets the data folder to store,
        would also check if it exists.
        '''
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

    def _check_existence(self, passed_path:str) -> bool:
        '''
        Instance method to check weather the data directory
        exists; otherwise return false
        passed_path(str): Desired path to check if it exists
        '''
        return os.path.exists(passed_path)



