import os
import hydra
from hydra.core.config_store import ConfigStore

from src.entities import TrainPipelineCfg


# TYPE HERE YOUR USERNAME AND KAGGLE KEY
os.environ['KAGGLE_USERNAME'] = "matthewiskornev"  # username
os.environ['KAGGLE_KEY'] = "52acdf9dcc5d73313cef38d121b1653e"  # key


from kaggle.api.kaggle_api_extended import KaggleApi


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def load(param: TrainPipelineCfg):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('cherngs/heart-disease-cleveland-uci', path=os.path.dirname(param.paths.path_to_raw_data), unzip=True)


def main():
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=TrainPipelineCfg)
    load()


if __name__ == '__main__':
    main()
