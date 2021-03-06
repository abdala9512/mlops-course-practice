"""End to end pipeline processing"""

from trainer import Trainer, HyperTuner
from utils import BaseLogger, ModelManager, ModelReport
from data_builder import DataBuilder

class Process(BaseLogger):


    def __init__(self):
        super().__init__()
        self.data_builder = DataBuilder()
        self.trainer = None
    def __run_end_to_end_pipeline(self):
        """End to end pipeline execution
        """

        self.logger.info("Starting End to end pipeline processing ")

        self.data_builder.execute()
        self.trainer      = HyperTuner(target='worldwide_gross')
        model  = self.trainer.tune()
        ModelManager(model=model).update()

        # Predictions report
        ModelReport(model=self.trainer).generate_report(obj_pred=model)

        self.logger.info("Model traning pipeline finished")

    
    def run(self):
        """Run process class"""
        self.__run_end_to_end_pipeline()