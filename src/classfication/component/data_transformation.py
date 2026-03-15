import os 
import sys  
import numpy as np  
import pandas as pd 
from src.classfication.logger import logging 
from src.classfication.exception import CustomException 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from src.classfication.utiles import * 
from dataclasses import dataclass
import pickle
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer



@dataclass
class DataTransformationConfig: 
    preporcess_data_path = os.path.join("artifacts","preporcess.pkl") 

class DataTranformation: 
    def __init__(self): 
        self.data_transformation_config = DataTransformationConfig()  
    
    def get_transformation_object(self):
        try:
            """
            this function is responsible for data transformation
            """
            num_col = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
            cat_col = ['quality'] 

            encode = OneHotEncoder() 
            scale= StandardScaler() 

            num_pipleine = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scale',scale)
                ]
            )
            logging.info(f"Numeric calumn : {num_pipleine} ")
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encode',encode)
                    
                ]
            ) 
            logging.info(f"Categorical Column : {cat_pipeline}") 

            preporcess = ColumnTransformer(
                [
                    ('num_pipeline',num_pipleine,num_col),
                    ('cat_pipeline',cat_pipeline,cat_col)
                ]
            )
            return preporcess 
        except Exception as e: 
           raise CustomException(e,sys) 
    
    def initiate_data_transformation(self,train_path,test_path): 
        try:
            train_df = pd.read_csv(train_path)
            logging.info("train data read successfully..")
            test_df = pd.read_csv(test_path) 
            logging.info("test data read successfully..") 

            preporcess_obj = self.get_transformation_object()  
            
            output_col = 'type'
            # divide the train dataset into input feature and output feature 
            input_feature_train= train_df.drop([output_col],axis=1)
            output_feature_train = train_df[output_col].map({'red':0,'white':1}) 

            #  divide the test dataset into  input feature and output feature.. 
            input_feature_test = test_df.drop([output_col],axis=1) 
            output_feature_test = test_df[output_col].map({'red':0,'white':1})

            #  preprocess 
            input_feature_train_arr = preporcess_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preporcess_obj.transform(input_feature_test) 

            train_arr = np.c_[input_feature_train_arr,np.array(output_feature_train)]
             
            test_arr = np.c_[input_feature_test_arr,np.array(output_feature_test)]
            
            os.makedirs(os.path.dirname(self.data_transformation_config.preporcess_data_path),exist_ok=True)

            save_object(file_path=self.data_transformation_config.preporcess_data_path,object=preporcess_obj)
            logging.info("Preprocess object save successully..")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preporcess_data_path
            ) 
    
        except Exception as e: 
            raise CustomException(e,sys)

