from fastapi import FastAPI
from fastapi_health import health
from sklearn.pipeline import Pipeline
import os
import pickle
import uvicorn
import logging

from microservice.functions import get_model_response
from microservice.entities import InputClass, OutputClass
from microservice.my_transformer import MyTransformer

app = FastAPI()

logger = logging.getLogger("logging_service")
logger.setLevel(logging.INFO)
formatter_stdout = logging.Formatter("%(asctime)s\t%(levelname)s\t%(name)20s\t%(message)s",
                                     datefmt="%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter_stdout)
logger.addHandler(stream_handler)

model_name = "Heart Disease Detector"
version = "v1.0.0"
model = None
transformer = None


@app.get("/")
def main() -> str:
    return "it is entry point of our service. Please add '/docs' to run prediction"


@app.get('/info')
async def model_info() -> dict:
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }


# class RenameUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         renamed_module = module
#         if module == "src":
#             renamed_module = "microservice"
#
#         return super(RenameUnpickler, self).find_class(renamed_module, name)
#
#
# def renamed_load(file_obj):
#     return RenameUnpickler(file_obj).load()


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


@app.on_event("startup")
def load_model():
    logger.info("Server started")
    global model
    global transformer
    model_path = os.getenv("PATH_TO_MODEL")
    trans_path = os.getenv("PATH_TO_TRANSFORMER")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.info(err)
        raise RuntimeError(err)
    if trans_path is None:
        err = f"PATH_TO_MODEL {trans_path} is None"
        logger.info(err)
        raise RuntimeError(err)

    model = load_object(model_path)
    transformer = load_object(trans_path)
    logger.info("Model was loaded successful")


def is_model_ready() -> bool:
    model_ready = not (model is None)
    transformer_ready = not (transformer is None)
    if not model_ready or not transformer_ready:
        logger.info('Model or transformer were not loaded')
    return model_ready and transformer_ready


async def success_handler(**kwargs) -> dict:
    return {'status': 'Model and transformer are ready'}


async def failure_handler(**kwargs) -> dict:
    return {'status': 'Model or transformer is not ready'}


app.add_api_route("/health", health([is_model_ready],
                                    success_status=200,
                                    success_handler=success_handler,
                                    failure_handler=failure_handler))


@app.post('/predict', response_model=OutputClass)
async def model_predict(input: InputClass) -> dict:
    """Predict with input"""
    response = get_model_response(input, model, transformer)
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
