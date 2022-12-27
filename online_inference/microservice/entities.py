from pydantic import BaseModel, validator, root_validator
from fastapi import HTTPException


FLOAT_FEATURES = ['oldpeak']
TRUE_ORDER_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


class InputClass(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    @validator('age')
    def check_age(cls, v):
        if v > 110 or v < 0:
            raise HTTPException(status_code=400, detail=f"You print incorrect age {v}.")
        return v

    @validator('sex')
    def check_sex(cls, v):
        if v not in [0, 1]:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect sex: {v}. Please input (1 = male; 0 = female).")
        return v

    @validator('cp')
    def check_cp(cls, v):
        correct_input = {'0': 'typical angina',
                         '1': 'atypical angina',
                         '2': 'non-anginal pain',
                         '3': 'asymptomatic'}
        if v not in [0, 1, 2, 3]:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect chest pain type: {v}. Please input {correct_input}.")
        return v

    @validator('trestbps')
    def check_trestbps(cls, v):
        if v > 240 or v < 80:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect resting blood pressure (in mm Hg on admission to the hospital) {v}.")
        return v

    @validator('chol')
    def check_chol(cls, v):
        if v > 600 or v < 0:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect serum cholestoral in mg/dl {v}.")
        return v

    @validator('fbs')
    def check_fbs(cls, v):
        if v not in [0, 1]:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect fasting blood sugar: {v}. Please input (1 = true; 0 = false).")
        return v

    @validator('restecg')
    def check_restecg(cls, v):
        correct_input = {'0': 'normal',
                         '1': 'having ST-T wave abnormality',
                         '2': 'showing probable or definite left ventricular hypertrophy by Estes criteria'}
        if v not in [0, 1, 2]:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect resting electrocardiographic results: {v}. Please input {correct_input}.")
        return v

    @validator('thalach')
    def check_thalach(cls, v):
        if v > 240 or v < 0:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect maximum heart rate achieved {v}.")
        return v

    @validator('exang')
    def check_exang(cls, v):
        if v not in [0, 1]:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect exercise induced angina: {v}. Please input (1 = yes; 0 = no).")
        return v

    @validator('oldpeak')
    def check_oldpeak(cls, v):
        if v > 7 or v < 0:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect ST depression induced by exercise relative to rest {v}.")
        return v

    @validator('slope')
    def check_slope(cls, v):
        correct_input = {'0': 'upsloping',
                         '1': 'flat',
                         '2': 'downsloping'}
        if v not in [0, 1, 2]:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect slope of the peak exercise ST segment: {v}. Please input {correct_input}.")
        return v

    @validator('ca')
    def check_ca(cls, v):
        if v not in [0, 1, 2, 3]:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect  number of major vessels (0-3) colored by flourosopy: {v}.")
        return v

    @validator('thal')
    def check_thal(cls, v):
        correct_input = {'0': 'normal',
                         '1': 'fixed defect',
                         '2': 'reversable defect'}
        if v not in [0, 1, 2]:
            raise HTTPException(
                status_code=400,
                detail=f"You print incorrect thal: {v}. Please input {correct_input}.")
        return v

    @root_validator
    def check_missing_features(cls, values):
        keys = list(values.keys())
        for i in range(len(TRUE_ORDER_FEATURES)):
            if TRUE_ORDER_FEATURES[i] not in keys:
                raise HTTPException(
                    status_code=400,
                    detail=f"You did not print feature {TRUE_ORDER_FEATURES[i]}. List of features {TRUE_ORDER_FEATURES}.")
        return values

    @root_validator(pre=True)
    def check_no_target(cls, values):
        if 'condition' in values:
            raise HTTPException(
                status_code=400,
                detail='target should not be included')
        return values

    @root_validator(pre=True)
    def check_order_features(cls, values):
        keys = list(values.keys())
        keys = tuple(keys)
        true_keys = tuple(TRUE_ORDER_FEATURES)
        if len(keys) != len(true_keys):
            return values
        if keys != true_keys:
            raise HTTPException(
                status_code=400,
                detail=f"You print features in wrong order or with mistakes. Correct order {TRUE_ORDER_FEATURES}.")
        return values


class OutputClass(BaseModel):
    label: str
    prediction: int
    probability: float
