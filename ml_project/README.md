**Homework №1**
==============================

## *Launch instructions*


### ***Installation***
Activate your env and then run in `ml_project/`:
~~~
pip install -r requirements.txt
pip install .
~~~
 


### ***Load data***
1. Go to file `ml_project/src/data/data_load.py` and type here\
   your kaggle username and key.
2. Go to `ml_project/` by typing `cd ..` two times in console.
3. To load data run:
    ~~~
    python src/data/data_load.py
    ~~~
4. The data loads to `ml_project/data/raw/heart_cleveland_upload.csv`.

### ***Make EDA report***
1. Go to `ml_project/`.
2. Run
   ~~~
   python reports/make_report.py
   ~~~
3. Report loads to `ml_project/eda_report.html`.


### ***Run mlflow***
1. Run this command:
   ~~~
   mlflow server --backend-store-uri sqlite:///:memory --default-artifact-root ./mlruns
   ~~~
2. Go to http address in console to enter mlflow in browser.
3. Switch to new window in console for further commands.

### ***Train model***
To use KNeighborsClassifier run in `ml_project/`:
~~~
python src/models/train_pipeline.py
~~~
To use RandomForestClassifier:
~~~
python src/models/train_pipeline.py model=rfc
~~~
If you want to use grid_search you should run this command in the following way:
~~~
python src/models/train_pipeline.py model.grid_search=true
python src/models/train_pipeline.py model=rfc model.grid_search=true
~~~


Processed train data with target column loads to `ml_project/data/processed/heart_cleveland_upload.csv`.\

Unprocessed test data and test target column loads to `ml_project/data/raw/feature_test.csv`\
`ml_project/data/raw/target_test.csv`\
respectively.

Fitted model loads to `ml_project/data/model/model.pkl`.


### ***Make predictions***
run in `ml_project/`:
~~~
python src/models/predict_pipeline.py
~~~
Predicted target loads to `ml_project/data/predicted_target.txt`

If you want to save predicted target in special file you should run this command with path to needed file.\
Example:
~~~
python src/models/predict_pipeline.py  -ppt=/Users/User/Documents/predicted_target.txt
~~~

If you want to use your train data instead of default (`ml_project/data/raw/feature_test.csv`)\
You should run this command with path to your data.
Example:
~~~
python src/models/predict_pipeline.py  -ptd=/Users/User/Documents/my_train_data.txt
~~~

### ***Tests***
1. Make synthetic data close to real by running in `ml_project/`:
   ~~~
   make_synth_data
   ~~~
   Data loads to `ml_project/tests/synth_data/synthetic_data.csv`
2. Run tests:
   ~~~
   python -m unittest tests/test_train_predict_model.py
   ~~~


## *Project organization*
    ├── build
    ├── data
    │   ├── model                <- Trained and serialized models
    │   ├── processed            <- The final, canonical data sets for modeling.
    │   ├── raw                  <- The original, immutable data dump.
    │   └── predicted_target.txt <- file for predicted target by model
    │
    ├── reports                  <- Directory for python scripts to generate html report
    │   ├── pictures             <- Graphics from eda notebook
    │   ├── templates            <- html scripts
    │   │   └── report_template.html
    │   └── make_report.py       <- python script
    │
    ├── src                      <- Source code for use in this project.
    │   ├── __init__.py          <- Makes ml_base_1 a Python module
    │   │
    │   ├── config               <- directiry with configs for project
    │   │   ├── hydra
    │   │   ├── model            <- directiry with configs for models
    │   │   └── config.yaml      <- general config
    │   │
    │   ├── data                 <- code to download data
    │   │   └── data_load.py
    │   │
    │   ├── entities             <- data class description
    │   │   └── train_params.py
    │   │
    │   ├── features             <- code to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models               <- code to train models and then use trained models to make predictions
    │   │   ├── model_fit_predict.py
    │   │   ├── predict_pipeline.py
    │   └── └── train_pipeline.py
    │
    ├── notebooks                <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   │                           the creator's initials, and a short `-` delimited description, e.g.
    │   │                           `1.0-jqp-initial-data-exploration`.
    │   └── 1.0-ime-initial-data-exploration.ipynb 
    │
    ├── tests                    <- Tests for code
    │   ├── synth_data           <- Synthetic data for testing
    │   ├── make_synth_data.py   <- Make synthetic data close to true
    │   └── test_train_predict_model.py
    │
    ├── definition.py
    ├── eda_report.py            <- eda report in html
    ├── README.md                <- Launch instructions and Project organization
    ├── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.
    │                               generated with `pip freeze > requirements.txt`
    │
    └── setup.py                 <- makes project pip installable (pip install -e .) so ml_base_1 can be imported