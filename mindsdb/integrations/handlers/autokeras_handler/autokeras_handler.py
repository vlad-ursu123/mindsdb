from typing import Optional
import os
import pandas as pd
import autokeras as ak
from mindsdb.integrations.libs.base import BaseMLEngine
from tensorflow.keras.models import load_model

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
    trainer.fit(training_df, df[target], epochs=DEFAULT_EPOCHS)
    return trainer.export_model()


def get_preds_from_autokeras_model(df, model, target):
    cols_to_drop = ["__mindsdb_row_id", target]
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
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

        
        model = train_autokeras_model(df, target)

        model.save(args["folder_path"])
        self.model_storage.json_set("predict_args", args)

    
    def predict(self, df, args=None):
        args = self.model_storage.json_get("predict_args")
        model = load_model(args["folder_path"], custom_objects=ak.CUSTOM_OBJECTS)

        df_to_predict = df.copy()
        predictions = get_preds_from_autokeras_model(df_to_predict, model, args["target"])
        df_to_predict[args["target"]] = predictions
        return df_to_predict
