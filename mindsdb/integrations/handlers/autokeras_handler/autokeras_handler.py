from typing import Optional
import os
import pandas as pd
import numpy as np
import autokeras as ak
from mindsdb.integrations.libs.base import BaseMLEngine
from tensorflow.keras.models import load_model

import logging

def configure_logger(log_file: str = "ml_handler_logs.txt") -> None:
    """Configure all the options for an app-wide logger.

    Parameters
    ----------
    log_file : str
        Path to the file where logs will be written to. File mode is set to append.
    """

    logging.basicConfig(
        format="%(asctime)s  %(levelname)s  %(message)s",
        filename=log_file,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="a+",
        level=logging.INFO,
    )
    return logging.getLogger("atlas-remote-logger")

def create_logger(log_file: str = "ml_handler_logs.txt", log_name: str = "mindsdb-logger-x") -> logging.Logger:
    """Configure and create a local logger.

    Returns
    -------
    logger : logging.Logger
        The pre-configured logger.
    """

    formatter = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    handler = logging.FileHandler(log_file, mode="a+")
    handler.setFormatter(formatter)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


logger = create_logger()



# Makes this run on Windows Subsystem for Linux
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/usr/lib/cuda"

trainer_dict = {
    "regression": ak.StructuredDataRegressor
}

DEFAULT_MODE = "regression"
DEFAULT_EPOCHS = 3
DEFAULT_TRIALS = 3

def train_autokeras_model(df, target, mode=DEFAULT_MODE):
    training_df = df.drop(target, axis=1)
    trainer = trainer_dict[mode](overwrite=True, max_trials=DEFAULT_TRIALS)

    logger.info("In train(): before training")
    logger.info(df)
    logger.info(target)
    logger.info(training_df)

    numeric_column_names = training_df.select_dtypes(include=[np.number]).columns.values.tolist()

    training_df = pd.get_dummies(training_df)
    logger.info("In train(): after getting dummies")
    logger.info(training_df)
    logger.info(training_df.columns.values.tolist())

    categorical_dummy_column_names = [col for col in training_df.columns.values.tolist() if col not in numeric_column_names]

    trainer.fit(training_df, df[target], epochs=DEFAULT_EPOCHS)
    return trainer.export_model(), categorical_dummy_column_names


def get_preds_from_autokeras_model(df, model, target, all_column_names):
    cols_to_drop = ["__mindsdb_row_id", target]
    for col in cols_to_drop:
        if col in df.columns.values.tolist():
            df = df.drop(col, axis=1)
    logger.info("In get_preds(): before getting dummies")
    logger.info(df)
    logger.info(target)
    logger.info("In get_preds(): after getting dummies")
    df = pd.get_dummies(df)
    logger.info(df)
    logger.info("In get_preds(): before filling missing columns")
    logger.info(all_column_names)
    logger.info(df.columns.values.tolist())
    for col in all_column_names:
        if col not in df.columns.values.tolist():
            df[col] = 0
    logger.info("In get_preds(): after filling missing columns")
    logger.info(df)

    return model.predict(df)



class AutokerasHandler(BaseMLEngine):
    """
    Integration with the AutoKeras ML library.
    """  # noqa

    name = 'autokeras'

    def create(self, target: str, df: Optional[pd.DataFrame] = None, args: Optional[dict] = None) -> None:
        """
        Create and train model on the input df
        """
        args = args['using']  # ignore the rest of the problem definition
        args["target"] = target
        args["folder_path"] = "autokeras"
        args["training_df"] = df.to_json()

        logger.info("In create(): Before training")
        logger.info(df)
        logger.info(df.dtypes)
        logger.info(df.shape)
        logger.info(target)
        args["training_data_column_count"] = len(df.columns) - 1 # subtract 1 for target
        model, args["data_column_names"] = train_autokeras_model(df, target)

        model.save(args["folder_path"])
        self.model_storage.json_set("predict_args", args)

    
    def predict(self, df, args=None):
        args = self.model_storage.json_get("predict_args")
        training_df = pd.read_json(args["training_df"])

        logger.info("Before load model")
        model = load_model(args["folder_path"], custom_objects=ak.CUSTOM_OBJECTS)

        df_to_predict = df.copy()
        if "__mindsdb_row_id" in df_to_predict.columns.values.tolist():
            df_to_predict = df_to_predict.drop("__mindsdb_row_id", axis=1)
            
        keys = list(df_to_predict.columns.values)
        i1 = training_df.set_index(keys).index
        i2 = df_to_predict.set_index(keys).index
        filtered_df = training_df[i1.isin(i2)]
        logger.info(filtered_df)
        logger.info(filtered_df.shape)
        logger.info("Before get predictions")

        predictions = get_preds_from_autokeras_model(filtered_df, model, args["target"], args["data_column_names"])
        filtered_df[args["target"]] = predictions
        return filtered_df
