import pandas as pd
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
#from src.components.data_ingestion import *
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformer_object(self):
        '''
        The Function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fixed syntax
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),  # Fixed syntax
                    ("cat_pipeline", cat_pipeline, categorical_columns)  # Fixed syntax
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving the preprocessed object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,  # Fixed attribute access
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,  # Fixed attribute access
            )

        except Exception as e:
            raise CustomException(e, sys)
