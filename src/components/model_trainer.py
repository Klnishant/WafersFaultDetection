import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import model_evaluation
from src.utils import load_object
from dataclasses import dataclass
import os
import sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_training_config=ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            X_train,X_test,y_train,y_test=(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )

            Models={
                'SVC':SVC(),
                'RandomForest':RandomForestClassifier(),
                'DecisionTree':DecisionTreeClassifier()
            }

            model_report:dict=model_evaluation(X_train,X_test,y_train,y_test,Models)
            print(model_report)
            print("\n==========================================================\n")
            logging.info(f'model_reoprt:{model_report}')

            best_model_score=max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = Models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_training_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e,sys)
