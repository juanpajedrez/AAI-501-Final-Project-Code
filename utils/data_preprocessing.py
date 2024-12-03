'''
Date: 2024-11-30
Group: 5
Course: AAI-501: Intro to AI
'''

#Lets import the necessary modules 
import pandas as pd
from category_encoders import TargetEncoder

class CreditCardPreprocesser:
    '''
    Contains all methods that perform
    data preprocessing on the credit card
    history
    '''
    def __init__(self, df_data:pd.DataFrame) -> None:
        '''
        Parameters (pd.DataFrame): Pandas dataframe of the credit card history
        containing all features, and target variable (is_fraud).
        '''
        self.df_data = df_data
    
    def fetch_preprocessed_dataframe(self) -> pd.DataFrame:
        '''
        Instance method that fetches the preprocessed dataframe
        of credit card history.
        '''
        self._drop_unique_identifiers()
        self._fill_nan_merch_zipcode()
        self._obtain_datetime_features()
        self._obtain_low_categorical_features()
        self._obtain_high_categorical_features()
        return self.df_data
    
    def _drop_unique_identifiers(self) -> None:
        '''
        Instance method that would drop the indices of
        unique indentifiers in the credit card history
        pandas dataframe, these were: trans_num and Unnamed: 0.
        '''
        self.df_data = self.df_data.drop("Unnamed: 0", axis=1)
        self.df_data = self.df_data.drop("trans_num", axis=1)
    
    def _fill_nan_merch_zipcode(self) -> None:
        '''
        Instance method that would fill nan placeholders
        in merch zipcode with median value
        '''
        self.df_data["merch_zipcode"] = self.df_data['merch_zipcode']\
            .fillna(self.df_data['merch_zipcode'].median())
    
    def _obtain_datetime_features(self) -> None:
        '''
        Instance method that would retrieve pre-processed features for
        datetime variables given in the credit card history. These were:
        transaction date time, and date of birth.
        '''
        self.df_data["trans_date_trans_time"] = pd.to_datetime(self.df_data["trans_date_trans_time"])
        self.df_data["dob"] = pd.to_datetime(self.df_data["dob"])

        self.df_data["transaction_year"] = self.df_data["trans_date_trans_time"].dt.year
        self.df_data["transaction_month"] = self.df_data["trans_date_trans_time"].dt.month
        self.df_data["transaction_day"] = self.df_data["trans_date_trans_time"].dt.day
        self.df_data["transaction_hour"] = self.df_data["trans_date_trans_time"].dt.hour
        self.df_data["transaction_minute"] = self.df_data["trans_date_trans_time"].dt.minute
        self.df_data["transaction_second"] = self.df_data["trans_date_trans_time"].dt.second

        self.df_data["birth_year"] = self.df_data["dob"].dt.year
        self.df_data["birth_month"] = self.df_data["dob"].dt.month
        self.df_data["birth_day"] = self.df_data["dob"].dt.day

        self.df_data.drop(["trans_date_trans_time", "dob"], axis=1, inplace = True)
    
    def _obtain_low_categorical_features(self) -> None:
        '''
        Instance method that would encode the low categorical features,
        as well as binary features, these are: Gender, category, state.
        '''
        self.df_data["gender"] = self.df_data["gender"].map({"F":1, "M": 0})
        self.df_data = pd.get_dummies(self.df_data,\
            columns=["category", "state"], drop_first=True, dtype=int)
    
    def _obtain_high_categorical_features(self) -> None:
        '''
        Instance method that would obtain the high categorical features
        using frequency encoding, as well as using a TargetEncoder()
        instance from scikit-learn.
        '''
        high_col_names = ["merchant", "first", "last", "street", "city", "job"]
        encoder = TargetEncoder()
        for c in high_col_names:
            #Create a name with _encoded and _freq
            col_enc_name = c +"_encoded"
            col_freq_name = c+ "_freq"

            #Use the target encoder to create the new column
            self.df_data[col_enc_name] = encoder.fit_transform(self.df_data[c], self.df_data["is_fraud"])

            #Obtain the frequency of the unique names for each
            freq = self.df_data[c].value_counts()
            self.df_data[col_freq_name] = self.df_data[c].map(freq)

            # Drop the original passed column
            self.df_data.drop(c, axis=1, inplace=True)