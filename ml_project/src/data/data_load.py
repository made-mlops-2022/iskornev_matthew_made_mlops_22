import os
from src.data import PATH_TO_DATA


# TYPE HERE YOUR USERNAME AND KAGGLE KEY
os.environ['KAGGLE_USERNAME'] = "matthewiskornev"  # username
os.environ['KAGGLE_KEY'] = "52acdf9dcc5d73313cef38d121b1653e"  # key


from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('cherngs/heart-disease-cleveland-uci', path=PATH_TO_DATA.joinpath('raw'), unzip=True)


if __name__ == '__main__':
    main()
