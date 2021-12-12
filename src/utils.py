"""Logger MLOps"""

import logging
import sys
import joblib

from sklearn.pipeline import Pipeline

class BaseLogger:

    def __init__(self):
        self.logger  = self.__logger_settings()


    def __logger_settings(self):

        logging.basicConfig(
            format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
            level=logging.INFO,
            datefmt='%H:%M:%S',
            stream=sys.stderr
        )
        logger =  logging.getLogger(__name__)

        return logger


class ModelUpdate(BaseLogger):

    def __init__(self, model:Pipeline) -> None:
        super().__init__()
        self.model = model

    def update(self) -> None:
        """Create .pkl file with the new trained model
        """
        self.logger.info("Updating models in models/model.pkl")
        joblib.dump(self.model, 'models/model.pkl')
