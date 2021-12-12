from fastapi import FastAPI
from .app.models import PredictionResponse, PredictionRequest
from .app.views import RequestHandler



class ModelService(FastAPI):

    def __init__(self) -> None:
        super().__init__()
        self.add_api_route(
            path = "/v1/prediction",
            endpoint = self._make_model_predictions,
            methods =['POST']
        )

    def _make_model_predictions(self, request: PredictionRequest):
        return PredictionResponse(
            worldwide_gross=RequestHandler.get_prediction(request)
            )

app = ModelService()