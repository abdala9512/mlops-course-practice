"""Utils MLOps"""

import logging
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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


class ModelManager(BaseLogger):

    def __init__(self,
        model:Pipeline=None,
        path=None) -> None:
        """Model manager initizalization
        """
        super().__init__()
        if model:
            self.model = model
        else:
            self.model = self.load(path)

    def update(self) -> None:
        """Creates .pkl file with the new trained model
        """
        self.logger.info("Updating models in models/model.pkl")
        joblib.dump(self.model, 'models/model.pkl')

    def load(self, path) -> Pipeline:
        """Returns loaded .pkl model
        """
        self.logger.info(f"Loading {path} binary...")
        return joblib.load(path)


class ModelReport(BaseLogger):


    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def generate_report(self, obj_pred):
        
        # Metrics report
        with open('model_report.txt', 'w') as report_file:
            report_file.write("# Model pipeline description ---------------------------- \n")

            for key, value in self.model.pipeline.named_steps.items():
                report_file.write(f" {key}:{value.__repr__()}" + "\n")

            report_file.write(f"## Train Score: {self.model.train_score} \n")
            report_file.write(f"## Test Score: {self.model.test_score} \n")

        # Plot report
        predictions = obj_pred.predict(self.model.X_test)

        self.logger.info("Plotting model results...")

        fig, ax = plt.subplots()
        fig.set_figheight(8)
        fig.set_figwidth(8)
        sns.regplot(x = predictions, y = self.model.y_test, ax = ax)

        ax.set_xlabel("Predicted worldwide gross")
        ax.set_ylabel("Observed worldwide gross")
        ax.set_title("GBM model predictions vs. observed")
        fig.savefig('predictions_movies.png')