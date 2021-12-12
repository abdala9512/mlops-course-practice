
from pydantic import BaseModel
import pandas as pd
import os 


def transform_to_dataframe(class_model: BaseModel) -> pd.DataFrame:
    transition_dictionary = {key:[value] for key, value in class_model.dict().items()}
    data_frame = pd.DataFrame(transition_dictionary)
    return data_frame