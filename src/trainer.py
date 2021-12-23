"""Gradient boost Machine trainer """

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from utils import BaseLogger
from typing import Tuple, Dict

import numpy as np
import pandas as pd

class Trainer(BaseLogger):
    """Trainer of Movies income predictions

    """

    def __init__(self, target, scorer='r2') -> None:
        """Initialize Trainer Class
        
        """
        super().__init__()
        self.data = None
        self.model_name = 'core_model'
        self.target = target
        self.scorer = scorer
        (
            self.data,
            self.features
        ) = self.__prepare_data(self.target)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = self.__split_data()
        self.pipeline = self.__create_pipeline()
        


    def __prepare_data(self, target):

        self.logger.info("Loading data...")
        data     = pd.read_csv("data/full_data.csv")
        features = data.drop(target, axis = 1).columns

        self.logger.info(f"Data target: {target}")
        self.logger.info(f"features: {' '.join(features)}")

        return data, features


    def __split_data(self):

        X = self.data[self.features]
        y = self.data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

        return X_train, X_test, y_train, y_test

    def __gbm_trainer(self, **params) -> GradientBoostingRegressor:
        """Instantiate a GBM regressor 
        """
        return GradientBoostingRegressor(**params)
        
         
    def __create_pipeline(self) -> Pipeline:
        """Create pipeline

        """
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean', missing_values=np.nan)),
            (self.model_name, self.__gbm_trainer())
        ])
        

    def fit(self):
        pass


class HyperTuner(Trainer):
    """Hyper parameter tuning of a Trainer Class
    
    """

    def __init__(self,target, scorer='r2'):
        super().__init__(target, scorer)

    def __instance_grid_search(self, model, grid) -> GridSearchCV:
        """Create an instance of  GridSearch object prepared to be fitted
        
        """
        return GridSearchCV(model, param_grid=grid, scoring=self.scorer, cv=5)

    def __generate_grid(self) -> Dict:

        return {
            f'{self.model_name}__n_estimators': range(20, 301, 20)
        }

    def tune(self):

        tuner = self.__instance_grid_search(
            model=self.pipeline,
            grid=self.__generate_grid()
            )

        self.logger.info("Starting grid Search...")
        tuner.fit(self.X_train, self.y_train)
        self.logger.info("Cross validating with best model...")

        final_model = cross_validate(
            tuner.best_estimator_,
            self.X_train, 
            self.y_train, 
            return_train_score=True
        )

        self.train_score = np.mean(final_model['train_score'])
        self.test_score  = np.mean(final_model['test_score'])

        self.logger.info(f"Train Score of optimized model: {self.train_score}")
        self.logger.info(f"Test Score of optimized model: {self.test_score}")

        return tuner.best_estimator_
