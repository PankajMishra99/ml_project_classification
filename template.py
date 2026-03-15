import os 
import logging 
from pathlib import Path

logging.basicConfig(level=logging.INFO) 
project_name = 'classfication'
list_of_files = [
    f"src/{project_name}/component/__init__.py",
    f"src/{project_name}/component/data_ingestion.py",
    f"src/{project_name}/component/data_transformation.py",
    f"src/{project_name}/component/model_trainer.py",
    f"src/{project_name}/component/data_monitoring.py", 
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/testing_pipeline.py", 
    f"src/{project_name}/logger.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/utiles.py",
    "artifacts/"
    "main.py",
    "app.py",
    "Dockerfile",
    "requirement.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath =Path(filepath)
    filedir, filename = os.path.split(filepath) 

    if filedir !='':
        os.makedirs(filedir,exist_ok=True) 
        logging.info(f"Creating directory for {filedir} and filename {filename}")
    
    if (not os.path.exists(filepath) or (os.path.getsize(filepath)==0)): 
        with open(filepath,'w') as file:
            pass 
            logging.info(f"Creating empty file {filename}")
    else: 
        logging.info(f"Filename {filename} has already exist..")
