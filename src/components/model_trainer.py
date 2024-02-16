import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model,SaveObject

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info ("Split training and trained data")
            X_train, Y_train, X_test, Y_test = (
                train_array[:,:-1], train_array[:,-1],
                test_array[:,:-1], test_array[:,-1]
            )

            #dictionary for models
            models = {
                "RandomForest":RandomForestRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "GradientBoosting":GradientBoostingRegressor(),
                "LinearRegression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostingRegressor":CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
            }

            params = {
                        "RandomForest":{
                            'n_estimators':[8,16,32,64,128, 256]
                        },
                        "DecisionTreeRegressor":{
                            'criterion':['squared_error', 'friedman_mse', 
                                 'absoulte_error', 'poisson']

                        },
                        "GradientBoosting":{
                            'learning_rate':[0.1,0.01,0.05, 0.001 ],
                            'subsample':[0.6,0.7,0.75, 0.8,0.85,0.9],
                            'n_estimators':[8,16,32,64,128, 256]
                        },
                        "LinearRegression":{},
                        "K-Neighbors Regressor":{
                            'n_neighbors':[5,7,9,11],
                        },
                        "XGBRegressor":{
                            'learning_rate':[0.01,0.05,0.1],
                            'n_estimators':[8,16,32,64,128, 256]
                        },
                        "CatBoostingRegressor":{
                            'depth': [6,8,10],
                            'learning_rate':[0.01,0.05,0.1],
                            'iterations':[30,50,100]
                        },
                        "AdaBoostRegressor": {
                            'learning_rate':[0.1,0.01, 0.5,0.001],
                            'n_estimators':[8,16,32,64,128, 256]
                        },
                    }
            

            #try hyperparameter tuning on your own ???
            model_report: dict = evaluate_model(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,models=models, params=params)

            ## to get best model from dict
            best_model_score = max(sorted(model_report.values()))
            print ("best score", best_model_score)
            ## to get best model name from dict
            best_model_name = list (model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if (best_model_score < 0.6):
                raise CustomException("No best model found")
            logging.info ("Best Model found with training and testing data")

            SaveObject (
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            Y_pred1=best_model.predict(X_test)
            r2_square = r2_score(Y_test,Y_pred1)
            return r2_square
        except Exception as e:
            raise CustomException (e, sys)    
        
