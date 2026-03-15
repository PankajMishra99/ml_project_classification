import os 
import sys 
import numpy as np  
import pandas as pd  
from src.classfication.logger import logging 
from src.classfication.exception import CustomException 
from dataclasses import dataclass 
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts",'raw_data.csv') 
    train_data_csv = os.path.join("artifacts",'train_data.csv')
    test_data_csv = os.path.join('artifacts','test_data.csv') 

class DataIngestion: 
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() 
    
    def initiate_data_config(self): 
        try: 
            df = pd.read_csv(os.path.join("notebooks",'wine_fraud.csv'))
            logging.info("raw data reading successfully..") 
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42) 
            train_data.to_csv(self.ingestion_config.train_data_csv,index=False,header = True)
            logging.info("train data saved sucessfully")
            test_data.to_csv(self.ingestion_config.test_data_csv,index=False,header=True) 
            logging.info("test data saved successfully..") 
            logging .info("Data ingestion is completed..") 
            return (
                self.ingestion_config.train_data_csv,
                self.ingestion_config.test_data_csv
            )
        except Exception as e:
            raise CustomException(e,sys)
