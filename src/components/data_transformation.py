import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data transformation
        
        '''
        try:
            numerical_columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']
            categorical_columns = [
                ['gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
                 'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
                 'PhoneService_Yes', 'MultipleLines_No',
                 'MultipleLines_No phone service', 'MultipleLines_Yes',
                 'InternetService_DSL', 'InternetService_Fiber optic',
                 'InternetService_No', 'OnlineSecurity_No',
                 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                 'OnlineBackup_No', 'OnlineBackup_No internet service',
                 'OnlineBackup_Yes', 'DeviceProtection_No',
                 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                 'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
                 'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
                 'StreamingMovies_No', 'StreamingMovies_No internet service',
                 'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
                 'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
                 'PaymentMethod_Bank transfer (automatic)',
                 'PaymentMethod_Credit card (automatic)',
                 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
                 'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36',
                 'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72']
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)