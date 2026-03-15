from src.classfication.utiles import * 
from src.classfication.component.data_ingestion import DataIngestion,DataIngestionConfig 
from src.classfication.component.data_transformation import DataTranformation,DataTransformationConfig 
from src.classfication.component.model_trainer import ModelTrainer,ModelTrainerConfig 
from src.classfication.logger import logging 
from src.classfication.exception import CustomException 

if __name__=="__main__": 
    logging.info("The excustion has started..") 

    try: 
        data_ingestion = DataIngestion()
        train_data_path,test_data_path= data_ingestion.initiate_data_config() 

        #  data transformation 
        data_transformation = DataTranformation()
        train_array,test_array,_ =data_transformation.initiate_data_transformation(train_data_path,test_data_path) 

        #  data trainer 
        model_trainer = ModelTrainer() 
        print(model_trainer.initiate_model_trainer(train_array,test_array)) 
    
    except Exception as e: 
        raise CustomException(e,sys)