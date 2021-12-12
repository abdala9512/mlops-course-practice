from .models import PredictionRequest
from .api_utils import transform_to_dataframe
from src.utils import ModelManager

model = ModelManager(path = 'models/model.pkl')

class RequestHandler:

    @classmethod
    def get_prediction(
        cls, 
        request: PredictionRequest
        ) -> float:
        
        data_to_predict = transform_to_dataframe(request)
        prediction = model.model.predict(data_to_predict)[0]
        return max(0, prediction)