from typing import List
from dataclasses import dataclass, field


@dataclass
class SplittingParams:
    test_size: float = field(default=0.3)
    random_state: int = field(default=42)


@dataclass
class FeatureList:
    categorical: List[str]
    numerical: List[str]
    target: str


@dataclass()
class ModelParams:
    name: str = field(default='knn')
    num_n: int = field(default=3)
    metric: str = field(default='minkowski')
    n_estimators: int = field(default=150)
    max_depth: int = field(default=5)
    max_features: str = field(default='log2')
    random_state: int = field(default=42)
    grid_search: bool = field(default=False)


@dataclass()
class KnnGridParams:
    n_neighbors: List[int]
    metric: List[str]


@dataclass()
class RfcGridParams:
    n_estimators: List[int]
    max_depth: List[int]
    max_features: List[str]


@dataclass()
class PathList:
    path_to_raw_data: str
    path_to_processed_data: str
    path_to_model: str
    path_to_transformer: str


@dataclass
class TrainPipelineCfg:
    model: ModelParams
    splitting_params: SplittingParams
    features: FeatureList
    knn_grid: KnnGridParams
    rfc_grid: RfcGridParams
    paths: PathList
