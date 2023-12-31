# training_pipeline.py

from src.CreditCardDefaultsPrediction.components.data_ingestion import DataIngestion
from src.CreditCardDefaultsPrediction.components.data_transformation import DataTransformation
from src.CreditCardDefaultsPrediction.components.model_trainer import ModelTrainer
from src.CreditCardDefaultsPrediction.components.model_evaluation import ModelEvaluation
from dataclasses import dataclass
import mlflow
import src.CreditCardDefaultsPrediction.utils.mlflow_setup as mlflow_setup
from urllib.parse import urlparse


@dataclass
class TrainingPipeline:

    def start(self):

        mlflow.set_registry_uri("https://dagshub.com/abhijitpaul0212/Credit-Card-Default-Detection.mlflow")
        mlflow_setup.start_mlflow_run()

        data_ingestion = DataIngestion()
        train_data_path, test_data_path, val_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, val_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path, val_data_path)

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, val_arr)

        model_evaluation = ModelEvaluation()
        model_evaluation.initiate_model_evaluation(test_arr)


if __name__ == '__main__':
    training_pipeline = TrainingPipeline()
    training_pipeline.start()
