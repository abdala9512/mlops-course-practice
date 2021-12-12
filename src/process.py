"""End to end pipeline processing"""

from trainer import Trainer, HyperTuner
from utils import BaseLogger, ModelUpdate
from data_builder import DataBuilder

class Process(BaseLogger):


    def __init__(self):
        super().__init__()
        self.data_builder = DataBuilder()
        self.trainer      = HyperTuner(target='worldwide_gross')

    def __run_end_to_end_pipeline(self):
        """End to end pipeline execution
        """

        self.logger.info("Starting End to end pipeline processing ")
        
        self.data_builder.execute()
        model  = self.trainer.tune()
        ModelUpdate(model=model).update()

    
    def run(self):
        """Run process class"""
        self.__run_end_to_end_pipeline()