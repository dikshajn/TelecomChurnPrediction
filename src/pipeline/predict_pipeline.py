import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
                 gender: str,
                 partner: str,
                 dependents: str,
                 phone_service: str,
                 multiple_lines: str,
                 internet_service: str,
                 online_security: str,
                 online_backup: str,
                 device_protection: str,
                 tech_support: str,
                 streaming_tv: str,
                 streaming_movies: str,
                 contract: str,
                 paperless_billing: str,
                 payment_method: str,
                 tenure_group: str):

        self.gender = gender
        self.partner = partner
        self.dependents = dependents
        self.phone_service = phone_service
        self.multiple_lines = multiple_lines
        self.internet_service = internet_service
        self.online_security = online_security
        self.online_backup = online_backup
        self.device_protection = device_protection
        self.tech_support = tech_support
        self.streaming_tv = streaming_tv
        self.streaming_movies = streaming_movies
        self.contract = contract
        self.paperless_billing = paperless_billing
        self.payment_method = payment_method
        self.tenure_group = tenure_group

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "Partner": [self.partner],
                "Dependents": [self.dependents],
                "PhoneService": [self.phone_service],
                "MultipleLines": [self.multiple_lines],
                "InternetService": [self.internet_service],
                "OnlineSecurity": [self.online_security],
                "OnlineBackup": [self.online_backup],
                "DeviceProtection": [self.device_protection],
                "TechSupport": [self.tech_support],
                "StreamingTV": [self.streaming_tv],
                "StreamingMovies": [self.streaming_movies],
                "Contract": [self.contract],
                "PaperlessBilling": [self.paperless_billing],
                "PaymentMethod": [self.payment_method],
                "tenure_group": [self.tenure_group],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
