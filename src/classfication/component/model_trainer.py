import os 
import sys  
import numpy as np  
import pandas as pd  
from src.classfication.logger import logging 
from src.classfication.exception import CustomException 
from src.classfication.utiles import * 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from dataclasses import dataclass 
from sklearn.metrics import *
import mlflow
@dataclass 
class ModelTrainerConfig:
    traing_config_path = os.path.join('artifacts','model.pkl') 

class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 
    
    def evaluate_metrics(self,actual,predict):
        prec = precision_score(actual,predict)
        recall = recall_score(actual,predict) 
        f1_sc  =f1_score(actual,predict) 
        return prec, recall,f1_sc 
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spliting data train and test input data..")
            x_train = train_array[:,:-1]
            y_train=train_array[:,-1] 
            x_test = test_array[:,:-1] 
            y_test = test_array[:,-1] 

            # define model
            models={
                    'logistic regresser': LogisticRegression(),
                    'random forest' : RandomForestClassifier(), 
                    'decision tree': DecisionTreeClassifier(),
                    'Neghbiour' : KNeighborsClassifier()
                    }
            
            #  parameter for the same 
            params ={
                'logistic regresser':{
                    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
                },
                'random forest':{
                    'criterion': ['gini', 'entropy', 'log_loss'],
                },
                'decision tree':{
                    'criterion': ['gini', 'entropy', 'log_loss']
                },
                'Neghbiour':{
                     'n_neighbors': [3, 5, 7, 9, 11],
                }

            } 
            model_report:dict = evaluate_model(x_train,x_test,y_train,y_test,models,params) 

            best_model_score = max(sorted(model_report.values())) 

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ] 
            logging.info(f"Best model name is {best_model_name}")

            best_model = models[best_model_name] 

            actual_model =''
            for model in list(models.keys()):
                if best_model_name==model: 
                    actual_model = actual_model + model 
            
            best_param = params[actual_model] 
            logging.info(f"Best model params is {best_param}")
            
            #  mlflow 
            mlflow.set_registry_uri("") 

            with mlflow.start_run(): 
                pred_value = best_model.predict(x_test) 
                prec, recall,f1_sc  = self.evaluate_metrics(y_test,pred_value) 

                mlflow.log_params(best_param) 
                mlflow.log_metric("precision",prec) 
                mlflow.log_metric('Recall',recall) 
                mlflow.log_metric('F1 scaore',f1_sc) 

                mlflow.sklearn.log_model("Model",best_model) 
            
            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info("Best model for both training and testing data..") 

            save_object(
                self.model_trainer_config.traing_config_path,
                object=best_model
            )

            predict_score  = model.predict(y_test)
            f1_sc = f1_score(y_test,predict_score) 
            return f1_sc 
        except Exception as e: 
            raise CustomException(e,sys) 
        

