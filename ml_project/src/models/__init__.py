import os
from pathlib import Path
from .model_fit_predict import train_model

__all__ = ['train_model']


tmp = os.path.abspath(os.curdir)
PATH_TO_MODEL = Path(tmp).joinpath('data/model')
