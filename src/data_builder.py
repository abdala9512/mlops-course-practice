"""Data preparation class"""

from dvc import api

import pandas as pd
from io import StringIO
import numpy as np

from utils import  BaseLogger

class DataBuilder(BaseLogger):

    def __init__(self):
        super().__init__()


    def __get_dvc_data(self):

        self.logger.info("Obtaining data from Google cloud storage...") 

        movies_path     = api.read("data/movies.csv", remote = 'data-dvc')
        finantials_path = api.read("data/finantials.csv", remote = 'data-dvc')
        gross_op_path   = api.read("data/opening_gross.csv", remote = 'data-dvc')


        movies     = pd.read_csv(StringIO(movies_path))
        finantials = pd.read_csv(StringIO(finantials_path))
        gross_op   = pd.read_csv(StringIO(gross_op_path))

        return movies, finantials, gross_op

    def __data_process(self):

        
        (
            movies, 
            finantials,
            gross_op 
        ) = self.__get_dvc_data()

        self.logger.info("Processing data...")

        numeric_columns_mask = (movies.dtypes == float) | (movies.dtypes == int)
        numeric_columns = [column for column in numeric_columns_mask.index if numeric_columns_mask[column]]
        movie_data = movies[numeric_columns+['movie_title']]

        fin_data = finantials[['movie_title', 'production_budget', 'worldwide_gross']]

        fin_movie_data = pd.merge(fin_data, movie_data, on='movie_title', how='left')
        full_movie_data = pd.merge(gross_op, fin_movie_data, on='movie_title', how='left')

        full_movie_data = full_movie_data.drop(['gross','movie_title'], axis=1)

        full_movie_data.to_csv('data/full_data.csv',index=False)

        self.logger.info('Data Fetched and prepared...')

    def execute(self):
        self.__data_process()