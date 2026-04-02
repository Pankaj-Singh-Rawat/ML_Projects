import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test data")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42, verbosity=0),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(random_state=42),
            }

            # Reduced + efficient param grid
            params = {
                "Decision Tree": {
                    "max_depth": [5, 10, None]
                },
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                },
                "CatBoost": {
                    "depth": [6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [50, 100]
                },
                "AdaBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                }
            }

            logging.info("Starting model evaluation")

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Best model selection
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No suitable model found", sys)

            # Train best model
            best_model.fit(X_train, y_train)

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info(f"Final R2 Score: {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)