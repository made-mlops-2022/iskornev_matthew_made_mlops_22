from pathlib import Path
import os


tmp = os.path.abspath(os.curdir)
PATH_TO_DATA = Path(tmp).joinpath('data')
